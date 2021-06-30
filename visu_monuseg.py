import matplotlib
import os
from os import path
from os.path import isfile,join
import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import sys
from dataset_utils import MoNuSegDataset
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import argparse
from argparse import ArgumentParser
from PIL import Image

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

def infere_and_save(model,save_dir,list_iter,test_dataset_no_norm,test_dataset,device,cpt):
    
    CMAP = U.get_cmap_landcover()
    for i in list_iter:
        path_save = join(save_dir,str(i))
        path_exist = path.isdir(path_save)
        

        if not path_exist: # Create Dir if not exists and save image and mask 
            os.mkdir(path_save)
            im,m = test_dataset_no_norm.__getitem__(i)
            im.transpose_(0,2)
            im.transpose_(0,1)
            im = im.numpy()
            m = m.numpy()
            im = im*255
            im = im.astype(np.uint8)
            im = Image.fromarray(im)
            m = Image.fromarray(m)
            im = im.convert("RGB")
            m = m.convert("L")
            m.save(join(path_save,'gt.png'))
            im.save(join(path_save,'image.png'))


        im,m = test_dataset.__getitem__(i)
        x = im.unsqueeze(0).to(device)
        pred = model(x)
        pred = pred['out']
        pred = pred.argmax(dim=1).squeeze().cpu()

        fig = plt.figure()
        plt.imshow(pred,cmap=CMAP,vmin=0,vmax=3,interpolation='nearest')
        plt.savefig(join(path_save,'pred'+str(cpt)+'.png'))
        
        

def main():
    #torch.manual_seed(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    

    parser.add_argument('--gpu', default=0, type=int,help="Device")
    args = parser.parse_args()



    # ------------
    # device
    # ------------
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    # ------------
    # model
    # ------------

    N_CLASSES = 4

    # Save_dir 

    save_dir = '/share/homes/karmimy/equiv/save_model/monuseg_visu'
    # ------------
    # dataset and dataloader
    # ------------
    dataroot_monuseg = '/share/DEEPLEARNING/datasets/monuseg'
    bs = 1
    num_classes = 2
    pm = True
    nw = 4
   

   
    
    entire_image = True
    print('Loading MoNuSeg Dataset')
    train_dataset = MoNuSegDataset(dataroot_monuseg,image_set='train',transforms=transforms_train,target_size=target_size,\
        stride=stride,binary=True,normalize=True)

    if entire_image:
        test_dataset = MoNuSegDataset(dataroot_monuseg,image_set='test',load_entire_image=True,binary=True)
        test_dataset_aji = MoNuSegDataset(dataroot_monuseg,image_set='test',load_entire_image=True,binary=False)
        test_dataset_no_norm = MoNuSegDataset(dataroot_monuseg,image_set='test',load_entire_image=True,binary=True,normalize=False)
    else:
        test_dataset = MoNuSegDataset(dataroot_monuseg,image_set='test',target_size=target_size,stride=stride,binary=True)
    print('Success load MoNuSeg Dataset')

    dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=nw,pin_memory=pm,\
    batch_size=bs)
    list_iter = np.arange(len(test_dataset))
    np.random.shuffle(list_iter)

    # count model 
    cpt = 0
    # First model
    model = torch.load('/share/homes/karmimy/equiv/save_model/fully_supervised_monuseg/2/fully_supervised_monuseg_ep49.pt',map_location=device)
    infere_and_save(model,save_dir,list_iter,test_dataset_no_norm,test_dataset,device,cpt)
    print('Visu of model',cpt,'Ended')
    aji,aji_mean = compute_AJI(model,dataloader_val,device,dist_factor=0.3,threshold=54,clean_prediction=False,it_bg=0,it_opening=0)
    model_save = 'aji_model0.txt'
    fi = os.path.join(save_dir,model_save)
    with open(fi,'w') as f:
        print(args,file=f)
    print('AJI of model 0 saved in',fi)
    cpt+=1

    model = torch.load('/share/homes/karmimy/equiv/save_model/fully_supervised_monuseg/6/fully_supervised_monuseg_ep49.pt',map_location=device)
    infere_and_save(model,save_dir,list_iter,test_dataset_no_norm,test_dataset,device,cpt)
    print('Visu of model',cpt,'Ended')
    aji,aji_mean = compute_AJI(model,dataloader_val,device,dist_factor=0.3,threshold=54,clean_prediction=False,it_bg=0,it_opening=0)
    model_save = 'aji_model1.txt'
    fi = os.path.join(save_dir,model_save)
    with open(fi,'w') as f:
        print(args,file=f)
    print('AJI of model 1 saved in',fi)
    cpt+=1
    
    model = torch.load('/share/homes/karmimy/equiv/save_model/rot_equiv_monuseg/6/rot_equiv_monuseg.pt',map_location=device)
    infere_and_save(model,save_dir,list_iter,test_dataset_no_norm,test_dataset,device,cpt)
    print('Visu of model',cpt,'Ended')
    aji,aji_mean = compute_AJI(model,dataloader_val,device,dist_factor=0.3,threshold=54,clean_prediction=False,it_bg=0,it_opening=0)
    model_save = 'aji_model2.txt'
    fi = os.path.join(save_dir,model_save)
    with open(fi,'w') as f:
        print(args,file=f)
    print('AJI of model 2 saved in',fi)


if __name__ == '__main__':
    main()



