from dataset_utils import MoNuSegDataset,split_dataset,Compose,RandomResize,RandomCrop,RandomPiRotate,\
    RandomHorizontalFlip,RandomRotate,CenterCrop,ToTensor,Normalize
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F 
from eval_train import train_fully_supervised
from argparse import ArgumentParser
import torch.utils.data as tud
import argparse
from eval_train import create_save_directory,save_hparams

# CONSTANTS


N_CLASSES = 3 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
    # Training parameters
    parser.add_argument('--auto_lr', type=str2bool, default=False,help="Auto lr finder")
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--scheduler', type=str2bool, default=False)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--benchmark', default=False, type=str2bool, help="enable or disable backends.cudnn")

    # Model and eval
    parser.add_argument('--model', default='FCN', type=str,help="FCN or DLV3 model")
    parser.add_argument('--pretrained', default=False, type=str2bool,help="Use pretrained pytorch model")
    parser.add_argument('--eval_angle', default=True, type=str2bool,help=\
        "If true, it'll eval the model with different angle input size")

    # Data augmentation 
    parser.add_argument('--rotate', default=False, type=str2bool,help="Use random rotation as data augmentation")
    parser.add_argument('--scale', default=True, type=str2bool,help="Use scale as data augmentation")
    parser.add_argument('--size_img', default=520, type=int,help="Size of input images")
    parser.add_argument('--size_crop', default=480, type=int,help="Size of crop image during training")
    parser.add_argument('--angle_max', default=360, type=int,help="Max angle for the random rotations")

    # Dataloader and gpu
    parser.add_argument('--nw', default=0, type=int,help="Num workers for the data loader")
    parser.add_argument('--pm', default=True, type=str2bool,help="Pin memory for the dataloader")
    parser.add_argument('--gpu', default=0, type=int,help="Wich gpu to select for training")
    
    # Dataset 
    parser.add_argument('--split', default=False, type=str2bool, help="Split the dataset")
    parser.add_argument('--split_ratio', default=0.3, type=float, help="Amount of data we used for training")
    parser.add_argument('--dataroot_monuseg', default='/share/DEEPLEARNING/datasets/m/', type=str)

    # Save parameters
    parser.add_argument('--model_name', type=str,help="what name to use for saving")
    parser.add_argument('--save_dir', default='/data/save_model', type=str)
    parser.add_argument('--save_all_ep', default=False, type=str2bool,help=\
        "If true it'll save the model every epoch in save_dir")
    parser.add_argument('--save_best', default=False, type=str2bool,help="If true will only save the best epoch model")
    args = parser.parse_args()
    # ------------
    # save
    # ------------
    save_dir = create_save_directory(args.save_dir)
    print('model will be saved in',save_dir)
    save_hparams(args,save_dir)
    # ------------
    # device
    # ------------
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    # ------------
    # model
    # ------------
    
    if args.model.upper()=='FCN':
        model = models.segmentation.fcn_resnet101(pretrained=args.pretrained,num_classes=N_CLASSES)
    elif args.model.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained,num_classes=N_CLASSES)
    else:
        raise Exception('model must be "FCN" or "DLV3"')
    model.to(device)
    # ------------
    # data augmentation
    # ------------
    if args.size_img < args.size_crop:
        raise Exception('Cannot have size of input images less than size of crop')
    size_img = (args.size_img,args.size_img)
    size_crop = (args.size_crop,args.size_crop)
    if args.scale:
        resize = 1.5
    else:
        resize = 1 
    if args.rotate:
        angle_max = args.angle_max 
    else:
        angle_max = 0

    transforms_train = Compose(
    RandomResize(min_size=size_img,max_size=size_img*resize),
    RandomCrop(size_crop),
    RandomHorizontalFlip(p=0.5),
    RandomRotate(angle_max=angle_max),
    Normalize(MEAN, STD),
    )

    transforms_test = Compose(
    Normalize(MEAN, STD),
    )

    # ------------
    # dataset and dataloader
    # ------------

    train_dataset = MoNuSegDataset(args.dataroot_monuseg,image_set='train',transforms=transforms_train)
    test_dataset = MoNuSegDataset(args.dataroot_monuseg,image_set='test',transforms=transforms_test)

    split = args.split
    if split==True:
        train_dataset = split_dataset(train_dataset,args.split_ratio)
    # Print len datasets
    print("There is",len(train_dataset),"images for training and",len(test_dataset),"for validation")
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.nw,\
        pin_memory=args.pm,shuffle=True,drop_last=True)#,collate_fn=U.my_collate)
    dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=args.nw,pin_memory=args.pm,\
        batch_size=args.batch_size)
    # Decide which device we want to run on
    

    # ------------
    # training
    # ------------
    # Auto lr finding
    #if args.auto_lr==True:

    criterion = nn.CrossEntropyLoss(ignore_index=N_CLASSES) # On ignore la classe border.
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.moment,weight_decay=args.wd)
    train_fully_supervised(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,\
        criterion=criterion,optimizer=optimizer,save_folder=save_dir,scheduler=args.scheduler,model_name=args.model_name,\
            benchmark=args.benchmark, save_best=args.save_best,save_all_ep=args.save_all_ep,device=device,num_classes=21)


    


if __name__ == '__main__':
    main()