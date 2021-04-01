from dataset_utils import MoNuSegDataset
import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F 
import eval_train as ev
from argparse import ArgumentParser
import torch.utils.data as tud


def main():
    #torch.manual_seed(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--auto_lr', type=U.str2bool, default=False,help="Auto lr finder")
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--scheduler', type=U.str2bool, default=False)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--model', default='FCN', type=str,help="FCN or DLV3 model")
    parser.add_argument('--pretrained', default=False, type=U.str2bool,help="Use pretrained pytorch model")
    parser.add_argument('--eval_angle', default=True, type=U.str2bool,help=\
        "If true, it'll eval the model with different angle input size")
    parser.add_argument('--rotate', default=False, type=U.str2bool,help="Use random rotation as data augmentation")
    parser.add_argument('--scale', default=True, type=U.str2bool,help="Use scale as data augmentation")
    parser.add_argument('--size_img', default=520, type=int,help="Size of input images")
    parser.add_argument('--size_crop', default=480, type=int,help="Size of crop image during training")
    parser.add_argument('--nw', default=0, type=int,help="Num workers for the data loader")
    parser.add_argument('--pm', default=True, type=U.str2bool,help="Pin memory for the dataloader")
    parser.add_argument('--gpu', default=0, type=int,help="Wich gpu to select for training")
    parser.add_argument('--benchmark', default=False, type=U.str2bool, help="enable or disable backends.cudnn")
    parser.add_argument('--split', default=False, type=U.str2bool, help="Split the dataset")
    parser.add_argument('--split_ratio', default=0.3, type=float, help="Amount of data we used for training")
    parser.add_argument('--dataroot_voc', default='/share/DEEPLEARNING/datasets/voc2012/', type=str)
    parser.add_argument('--dataroot_sbd', default='/share/DEEPLEARNING/datasets/sbd/', type=str)
    parser.add_argument('--model_name', type=str,help="what name to use for saving")
    parser.add_argument('--save_dir', default='/data/save_model', type=str)
    parser.add_argument('--save_all_ep', default=False, type=U.str2bool,help=\
        "If true it'll save the model every epoch in save_dir")
    parser.add_argument('--save_best', default=False, type=U.str2bool,help="If true will only save the best epoch model")
    args = parser.parse_args()
    # ------------
    # save
    # ------------
    save_dir = U.create_save_directory(args.save_dir)
    print('model will be saved in',save_dir)
    U.save_hparams(args,save_dir)
    # ------------
    # device
    # ------------
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    # ------------
    # model
    # ------------
    
    if args.model.upper()=='FCN':
        model = models.segmentation.fcn_resnet101(pretrained=args.pretrained)
    elif args.model.upper()=='DLV3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained)
    else:
        raise Exception('model must be "FCN" or "DLV3"')
    model.to(device)
    # ------------
    # data
    # ------------
    if args.size_img < args.size_crop:
        raise Exception('Cannot have size of input images less than size of crop')
    size_img = (args.size_img,args.size_img)
    size_crop = (args.size_crop,args.size_crop)
    train_dataset_VOC = mdset.VOCSegmentation(args.dataroot_voc,year='2012', image_set='train', \
        download=True,rotate=args.rotate,scale=args.scale,size_img=size_img,size_crop=size_crop)
    val_dataset_VOC = mdset.VOCSegmentation(args.dataroot_voc,year='2012', image_set='val', download=True)
    train_dataset_SBD = mdset.SBDataset(args.dataroot_sbd, image_set='train_noval',mode='segmentation',\
        rotate=args.rotate,scale=args.scale,size_img=size_img,size_crop=size_crop)
    # Concatene dataset
    train_dataset = tud.ConcatDataset([train_dataset_VOC,train_dataset_SBD])
    split = args.split
    if split==True:
        train_dataset = U.split_dataset(train_dataset,args.split_ratio)
    # Print len datasets
    print("There is",len(train_dataset),"images for training and",len(val_dataset_VOC),"for validation")
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.nw,\
        pin_memory=args.pm,shuffle=True,drop_last=True)#,collate_fn=U.my_collate)
    dataloader_val = torch.utils.data.DataLoader(val_dataset_VOC,num_workers=args.nw,pin_memory=args.pm,\
        batch_size=args.batch_size)
    # Decide which device we want to run on
    

    # ------------
    # training
    # ------------
    # Auto lr finding
    #if args.auto_lr==True:

    criterion = nn.CrossEntropyLoss(ignore_index=21) # On ignore la classe border.
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.moment,weight_decay=args.wd)
    ev.train_fully_supervised(model=model,n_epochs=args.n_epochs,train_loader=dataloader_train,val_loader=dataloader_val,\
        criterion=criterion,optimizer=optimizer,save_folder=save_dir,scheduler=args.scheduler,model_name=args.model_name,\
            benchmark=args.benchmark, save_best=args.save_best,save_all_ep=args.save_all_ep,device=device,num_classes=21)

    # Final evaluation
    if args.eval_angle:
        d_iou = ev.eval_model_all_angle(model,args.size_img,args.dataroot_voc,train=True,device=device)
        U.save_eval_angle(d_iou,save_dir)
        d_iou = ev.eval_model_all_angle(model,args.size_img,args.dataroot_voc,train=False,device=device)
        U.save_eval_angle(d_iou,save_dir)
    


if __name__ == '__main__':
    main()