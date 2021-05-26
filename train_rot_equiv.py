from dataset_utils import MoNuSegDataset,split_dataset,Compose,RandomResize,RandomCrop,RandomPiRotate,\
    RandomHorizontalFlip,RandomRotate,CenterCrop,ToTensor,Normalize,RandomAffine
import torch
from torchvision import models
import torch.nn as nn
from eval_train import train_rot_equiv
from argparse import ArgumentParser
import torch.utils.data as tud
import argparse
from eval_train import create_save_directory,save_hparams
from model import CNN3
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
    # Training parameters
    parser.add_argument('--auto_lr', type=str2bool, default=False,help="Auto lr finder")
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--benchmark', default=True, type=str2bool, help="enable or disable backends.cudnn")
    parser.add_argument('--num_classes', default=3, type=int,help="How many classes for the model")

    # Model and eval
    parser.add_argument('--model', default='FCN', type=str,help="FCN or DLV3 model")
    parser.add_argument('--pretrained', default=False, type=str2bool,help="Use pretrained pytorch model")
    parser.add_argument('--eval_angle', default=False, type=str2bool,help=\
        "If true, it'll eval the model with different angle input size")
    parser.add_argument('--gamma', type=float, default=0.5,help="Loss balancing parameters: gamma*loss_equiv + (1-gamma)*loss_sup")
    parser.add_argument('--eval_every', default=30, type=int,help="Eval all input rotation angle every n step")
    parser.add_argument('--multi_task', default=False, type=str2bool,help="Multi task training")
    parser.add_argument('--Loss', default='KL', type=str,help="Loss used for equivariance cost function (KL or CE)")
    # Data augmentation 
    parser.add_argument('--rotate', default=False, type=str2bool,help="Use random rotation as data augmentation")
    parser.add_argument('--scale', default=True, type=str2bool,help="Use scale as data augmentation")
    parser.add_argument('--size_img', default=100, type=int,help="Size of input images")
    parser.add_argument('--size_crop', default=80, type=int,help="Size of crop image during training")
    parser.add_argument('--angle_max', default=360, type=int,help="Max angle for the random rotations")
    parser.add_argument('--pi_rotate', default=False, type=str2bool,help="Use only pi/2 multiple rotation for rotation equivariance training")
    # Dataloader and gpu
    parser.add_argument('--nw', default=0, type=int,help="Num workers for the data loader")
    parser.add_argument('--pm', default=True, type=str2bool,help="Pin memory for the dataloader")
    parser.add_argument('--gpu', default=0, type=int,help="Wich gpu to select for training")
    parser.add_argument('--rot_cpu', default=False, type=str2bool, help="Apply rotation on the cpu (Help to use less gpu memory)")
    
    # Dataset 
    parser.add_argument('--split', default=False, type=str2bool, help="Split the dataset")
    parser.add_argument('--split_ratio', default=0.3, type=float, help="Amount of data we used for training")
    parser.add_argument('--dataroot_monuseg', default='/share/DEEPLEARNING/datasets/monuseg', type=str)
    parser.add_argument('--entire_image', default=False, type=str2bool,help="Eval on the complete image 1000x1000 size")
    parser.add_argument('--target_size',default=None,type=str,help='Target size used for spliting images.')
    parser.add_argument('--stride',default=None,type=str,help='Stride used for spliting images.')
    # Save parameters
    parser.add_argument('--model_name', type=str,default="rot_equiv_monuseg",help="what name to use for saving")
    parser.add_argument('--save_dir', default='/share/homes/karmimy/equiv/save_model/rot_equiv_monuseg', type=str)
    parser.add_argument('--save_all_ep', default=False, type=str2bool,help=\
        "If true it'll save the model every epoch in save_dir")
    parser.add_argument('--save_best', default=True, type=str2bool,help="If true will only save the best epoch model")
    args = parser.parse_args()
    # ------------
    # device
    # ------------
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    # ------------
    # model
    # ------------

    N_CLASSES = args.num_classes
    
    if args.model.upper()=='FCN':
        model = models.segmentation.fcn_resnet101(pretrained=args.pretrained,num_classes=N_CLASSES)
    elif args.model.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained,num_classes=N_CLASSES)
    elif args.model.upper()=='CNN3':
        print('CAREFUL! If you use the model CNN3, the input size MUST BE 51.')
        model = CNN3()
    else:
        raise Exception('model must be "FCN" , "CNN3" or "DLV3"')
    model.to(device)
    # ------------
    # data augmentation
    # ------------
    if args.size_img < args.size_crop:
        raise Exception('Cannot have size of input images less than size of crop')
    size_img = args.size_img
    size_crop = args.size_crop
    if args.scale:
        min_size = 0.7
        resize = 1.3
        size_max=int(size_img*resize)
        size_min = size_crop
        size_img = size_min
        
    else:
        resize = 1
        size_max=size_max=size_img*resize

    if args.rotate:
        transforms_train = Compose([
        RandomResize(min_size=size_img,max_size=size_max),
        #RandomRotate(angle_max=args.angle_max,p_rotate=0.25,expand=True),
        RandomPiRotate(p_rotate=0.25),
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
        RandomAffine(p=0.25,angle=40,translate=(0.25,0.5),scale=1.5,shear=(-45.0,45.0))
        ])
        


    # ------------
    # dataset and dataloader
    # ------------

    train_dataset_unsup = MoNuSegDataset(args.dataroot_monuseg,image_set='train',transforms=transforms_train,target_size=args.target_size,stride=args.stride)
    if args.entire_image:
        test_dataset = MoNuSegDataset(args.dataroot_monuseg,image_set='test',load_entire_image=True)
        test_dataset_aji = MoNuSegDataset(args.dataroot_monuseg,image_set='test',load_entire_image=True,binary=False)
    else:
        test_dataset = MoNuSegDataset(args.dataroot_monuseg,image_set='test',target_size=args.target_size,stride=args.stride)

    

    # Split dataset
    split = args.split
    if split==True:
        train_dataset_sup = split_dataset(train_dataset_unsup,args.split_ratio)
    else:
        train_dataset_sup = train_dataset_unsup
    # Multi task ? 
    if args.multi_task: 
        train_dataset_unsup = train_dataset_sup
    # Print len datasets
    print("There is",len(train_dataset_unsup),"images for equivariance training,",len(train_dataset_sup),"for supervised training and",\
        len(test_dataset),"for validation")
    dataloader_train_sup = torch.utils.data.DataLoader(train_dataset_sup, batch_size=args.batch_size,num_workers=args.nw,\
        pin_memory=args.pm,shuffle=True,drop_last=True)

    if args.entire_image:
        dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=args.nw,pin_memory=args.pm,\
            batch_size=1) # Batch size set to 1 if we evaluate on the entire image (1000 x 1000 size)
        dataloader_val_aji = torch.utils.data.DataLoader(test_dataset_aji,num_workers=args.nw,pin_memory=args.pm,\
            batch_size=1)
    else:
        dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=args.nw,pin_memory=args.pm,\
            batch_size=args.batch_size)
        
    # Decide which device we want to run on
    

    
    # Auto lr finding
    #if args.auto_lr==True:
    # ------------
    # save
    # ------------
    save_dir = create_save_directory(args.save_dir)
    print('model will be saved in',save_dir)
    save_hparams(args,save_dir)
    print('PARAMETERS : ')
    print(args)
    print('-------------------------------------------------------------------')
    # ------------
    # training
    # ------------
    print('N_CLASSES',N_CLASSES)
    criterion_supervised = nn.CrossEntropyLoss(ignore_index=N_CLASSES) # On ignore la classe border.
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.moment,weight_decay=args.wd)

    train_rot_equiv(model,args.n_epochs,dataloader_train_sup,train_dataset_unsup,dataloader_val,criterion_supervised,optimizer,\
        scheduler=args.scheduler,Loss=args.Loss,gamma=args.gamma,batch_size=args.batch_size,save_folder=save_dir,\
            model_name=args.model_name,benchmark=args.benchmark,angle_max=args.angle_max,pi_rotate=args.pi_rotate,AJI=False,\
                aji_loader=dataloader_val_aji,eval_every=args.eval_every,save_all_ep=args.save_all_ep,save_best=args.save_best\
            ,rot_cpu=args.rot_cpu,device=device,num_classes=N_CLASSES)


if __name__ == '__main__':
    main()