from dataset_utils.my_transforms import ColorJitter
from dataset_utils import MoNuSegDataset,split_dataset,Compose,RandomResize,RandomCrop,RandomPiRotate,\
    RandomHorizontalFlip,RandomRotate,CenterCrop,ToTensor,Normalize,RandomAffine
import torch
from torchvision import models
import torch.nn as nn
from eval_train import train_fully_supervised
from argparse import ArgumentParser
import torch.utils.data as tud
import argparse
from eval_train import create_save_directory,save_hparams,compute_AJI
from model import CNN3
import yaml
from os.path import join
# CONSTANTS



### TYPE 
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def main():
    #torch.manual_seed(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    
    # Model and eval
    parser.add_argument('--config', default='config/fully_sup_config.yaml', type=str,help="Yaml configuration files")
    args = parser.parse_args()


    #------------
    # YAML
    #------------


    with open(args.config) as f:
        
        arguments = yaml.load(f, Loader=yaml.FullLoader)
        
    # Training args 
    auto_lr = arguments['training']['auto_lr']
    learning_rate = arguments['training']['learning_rate']
    scheduler = arguments['training']['scheduler']
    wd = arguments['training']['wd']
    moment = arguments['training']['moment']
    batch_size = arguments['training']['batch_size']
    benchmark = arguments['training']['benchmark']
    num_classes = arguments['training']['num_classes']
    n_epochs = arguments['training']['n_epochs']

    # Model args 
    model_n = arguments['model_config']['model']
    eval_angle = arguments['model_config']['eval_angle']
    pretrained = arguments['model_config']['pretrained']
    aji = arguments['model_config']['aji']
    # Data augmentation
    rotate = arguments['data_augmentation']['rotate']
    scale = arguments['data_augmentation']['scale']
    size_img = arguments['data_augmentation']['size_img']
    size_crop = arguments['data_augmentation']['size_crop']
    angle_max = arguments['data_augmentation']['angle_max']

    # Dataloader and gpu
    nw =  arguments['loader_gpu']['nw']
    pm =  arguments['loader_gpu']['pm']
    gpu =  arguments['loader_gpu']['gpu']

    # Datasets 
    split = arguments['dataset']['split']
    split_ratio = arguments['dataset']['split_ratio']
    dataroot_monuseg = arguments['dataset']['dataroot_monuseg']
    entire_image = arguments['dataset']['entire_image']
    target_size = arguments['dataset']['target_size']
    stride = arguments['dataset']['stride']

    # Save config 
    model_name = arguments['save_config']['model_name']
    save_dir = arguments['save_config']['save_dir']
    save_all_ep = arguments['save_config']['save_all_ep']
    save_best = arguments['save_config']['save_best']


    # ------------
    # device
    # ------------
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    # ------------
    # model
    # ------------

    N_CLASSES = num_classes
    
    if model_n.upper()=='FCN':
        model = models.segmentation.fcn_resnet101(pretrained=pretrained,num_classes=N_CLASSES)
    elif model_n.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained,num_classes=N_CLASSES)
    elif model_n.upper()=='CNN3':
        print('CAREFUL! If you use the model CNN3, the input size MUST BE 51.')
        model = CNN3()
    else:
        raise Exception('model must be "FCN" , "CNN3" or "DLV3"')
    model.to(device)
    # ------------
    # data augmentation
    # ------------
    if size_img < size_crop:
        raise Exception('Cannot have size of input images less than size of crop')

    if scale:
        min_size = 0.7
        resize = 1.3
        size_max=int(size_img*resize)
        size_min = size_crop
        size_img = size_min
        
    else:
        resize = 1
        size_max=size_max=size_img*resize

    jitter = 10
    if rotate:
        transforms_train = Compose([
        RandomResize(min_size=size_img,max_size=size_max),
        RandomRotate(angle_max=angle_max,p_rotate=0.25,expand=True),
        #RandomPiRotate(p_rotate=0.25),
        RandomCrop(size_crop),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomAffine(p=0.25,angle=40,translate=(0.25,0.5),scale=1.5,shear=(-45.0,45.0))
        ]
        )
    else:
        transforms_train = Compose([
        RandomResize(min_size=size_img,max_size=size_max),
        RandomCrop(size_crop),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomAffine(p=0.25)
        #ColorJitter(brightness=10,contrast=10,saturation=10)
        ])
        


    # ------------
    # dataset and dataloader
    # ------------

    train_dataset = MoNuSegDataset(dataroot_monuseg,image_set='train',transforms=transforms_train,target_size=target_size,\
        stride=stride,binary=True,normalize=True)
    if entire_image:
        test_dataset = MoNuSegDataset(dataroot_monuseg,image_set='test',load_entire_image=True,binary=True)
        test_dataset_aji = MoNuSegDataset(dataroot_monuseg,image_set='test',load_entire_image=True,binary=False)
    else:
        test_dataset = MoNuSegDataset(dataroot_monuseg,image_set='test',target_size=target_size,stride=stride,binary=True)

    

    split = split
    if split==True:
        train_dataset = split_dataset(train_dataset,split_ratio)
    # Print len datasets
    print("There is",len(train_dataset),"images for training and",len(test_dataset),"for validation")
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=nw,\
        pin_memory=pm,shuffle=True,drop_last=True)

    if entire_image:
        dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=nw,pin_memory=pm,\
            batch_size=1) # Batch size set to 1 if we evaluate on the entire image (1000 x 1000 size)
        dataloader_val_aji = torch.utils.data.DataLoader(test_dataset_aji,num_workers=nw,pin_memory=pm,\
            batch_size=1)
    else:
        dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=nw,pin_memory=pm,\
            batch_size=batch_size)

        
    # Decide which device we want to run on
    

    
    # Auto lr finding
    #if auto_lr==True:
    # ------------
    # save
    # ------------
    save_dir = create_save_directory(save_dir)
    print('model will be saved in',save_dir)
    save_hparams(args,save_dir)
    print('PARAMETERS : ')
    print(arguments)
    print('-------------------------------------------------------------------')
    # ------------
    # training
    # ------------
    print('N_CLASSES',N_CLASSES)
    criterion = nn.CrossEntropyLoss(ignore_index=N_CLASSES) # On ignore la classe border.
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=moment,weight_decay=wd)
    
    train_fully_supervised(model=model,n_epochs=n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,aji_loader=dataloader_val_aji,\
        criterion=criterion,optimizer=optimizer,save_folder=save_dir,scheduler=scheduler,model_name=model_name,\
            benchmark=benchmark,AJI=aji, save_best=save_best,save_all_ep=save_all_ep,device=device,num_classes=N_CLASSES)

    model = torch.load(join(save_dir,model_name),map_location=device)
    l_angles = [180,210,240,270,300,330,0,30,60,90,120,150]
    l_iou = []
    for angle in l_angles:
        test_dataset_aji = MoNuSegDataset(dataroot_monuseg,image_set='test',load_entire_image=True,binary=False,fixing_rotate=True,angle_fix=angle)
        dataloader_val = torch.utils.data.DataLoader(test_dataset_aji,num_workers=nw,pin_memory=pm,\
            batch_size=1)
        aji,aji_mean = compute_AJI(model,dataloader_val,device,dist_factor=0.3,threshold=54,clean_prediction=False,it_bg=0,it_opening=0)
        print('EVAL FOR ANGLE',angle,': AJI',aji_mean)


if __name__ == '__main__':
    main()