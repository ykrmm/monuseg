import os
from os.path import join
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

def to_tensor_target_lc(mask):
    # For the landcoverdataset
    mask = np.array(mask)
    return torch.LongTensor(mask)

class MoNuSegDataset(Dataset):
    def __init__(self,
                 dataroot,
                 image_set='train',
                 transforms = None,
                 fixing_rotate = False,
                 angle_fix = 0):
        super(MoNuSegDataset).__init__()

        ## Transform

        self.fixing_rotate = fixing_rotate # For test time, allow to eval a model on the dataset with a certain angle
        self.angle_fix = angle_fix # The angle used for the fixing rotation
        self.transforms = transforms

        ##
        image_set = image_set.lower()
        if image_set!= 'train' and image_set!='test':
            raise Exception("image set should be 'train' or 'test',not",image_set)
        self.train = image_set == 'train' 

        if self.train : 
            dataroot = join(dataroot,'MoNuSegTrainingData')
        else:
            dataroot = join(dataroot,'MoNuSegTestData')
        self.root_data = join(dataroot,'Output')
        list_file = join(dataroot, 'list.txt')
        with open(os.path.join(list_file), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

    def my_transform(self, image, mask):
        # Apply a fixed rotation for test time:
        if self.fixing_rotate:
            image = TF.rotate(image,angle=self.angle_fix)
            mask = TF.rotate(mask,angle=self.angle_fix)
        
        image,mask = self.transforms
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = to_tensor_target_lc(mask)
        return image, mask


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_img,self.file_names[index]+'.jpg')).convert('RGB') # Convert RGB ? 
        target = Image.open(os.path.join(self.root_img,self.file_names[index]+'_m.png'))
        img, target = self.my_transform(img, target)
        return img, target

    def __len__(self):
        return len(self.file_names)