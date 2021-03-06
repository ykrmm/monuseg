import os
from os.path import join
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

def to_tensor_target(mask,binary=True):
    mask = np.array(mask)
    if binary: 
        mask = np.where(mask > 1, 1, mask) # Binarise the mask  
    return torch.LongTensor(mask)

class MoNuSegDataset(Dataset):
    def __init__(self,
                 dataroot,
                 image_set='train',
                 transforms = None,
                 fixing_rotate = False,
                 angle_fix = 0,
                 mean = (0.485, 0.456, 0.406),
                 std  = (0.229, 0.224, 0.225),
                 normalize = True,
                 load_entire_image = False,
                 target_size=None,
                 stride = None,
                 binary = True):
        super(MoNuSegDataset).__init__()

        ## Transform
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.fixing_rotate = fixing_rotate # For test time, allow to eval a model on the dataset with a certain angle
        self.angle_fix = angle_fix # The angle used for the fixing rotation
        self.transforms = transforms
        self.binary = binary

        ## File loading 
        self.load_entire_image = load_entire_image

        image_set = image_set.lower()
        if image_set!= 'train' and image_set!='test':
            raise Exception("image set should be 'train' or 'test',not",image_set)
        self.train = image_set == 'train' 

        if self.train : 
            dataroot = join(dataroot,'MoNuSegTrainingData')
        else:
            dataroot = join(dataroot,'MoNuSegTestData')
        if target_size is not None and stride is not None:
            self.root_data = join(dataroot,'Output_'+target_size+'_'+stride)
        else: 
            self.root_data = join(dataroot,'Output')

        if self.load_entire_image:
            list_file = join(dataroot, 'list_entire_patch.txt')
            self.root_img = join(dataroot,'Tissue_Images')
            if self.binary:
                self.root_masks = join(dataroot,'Binary_masks')
            else:
                self.root_masks = join(dataroot,'Binary_masks_instance')
        else:
            if target_size is not None and stride is not None:
                list_file = join(dataroot, 'list_'+target_size+'_'+stride+'.txt')
            else: 
                list_file = join(dataroot, 'list.txt')
        with open(os.path.join(list_file), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

    def my_transform(self, image, mask):
        mask = to_tensor_target(mask,binary=self.binary)
        # Apply a fixed rotation for test time:
        if self.fixing_rotate:
            image = TF.rotate(image,angle=self.angle_fix,expand=True)
            mask = torch.unsqueeze(mask,0)
            mask = TF.rotate(mask,angle=self.angle_fix,expand=True)
            mask = mask.squeeze()
            
        
        if self.transforms is not None:
            image,mask = self.transforms(image,mask)
        # Transform to tensor
        image = TF.to_tensor(image)
        if self.normalize:
            image = TF.normalize(image,self.mean,self.std)
        
        return image, mask


    def __getitem__(self, index):
        if self.load_entire_image:
            img = Image.open(os.path.join(self.root_img,self.file_names[index]+'.tif')).convert('RGB') # Convert RGB ? 
            if self.binary:
                target = Image.open(os.path.join(self.root_masks,self.file_names[index]+'.png')).convert('L') # To have (h,w) images 
            else:
                target = np.load(os.path.join(self.root_masks,self.file_names[index]+'.npy'))
        else:
            img = Image.open(os.path.join(self.root_data,self.file_names[index]+'.jpg')).convert('RGB') # Convert RGB ? 
            target = Image.open(os.path.join(self.root_data,self.file_names[index]+'_m.png')).convert('L') # To have (h,w) images 
        img, target = self.my_transform(img, target)
        return img, target

    def __len__(self):
        return len(self.file_names)

