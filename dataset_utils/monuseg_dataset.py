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
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225],
                 size_img = (512,512),
                 size_crop = (480,480),
                 scale_factor = (0.5,1.2),
                 p = 0.5,
                 p_rotate = 0.25,
                 rotate = False,
                 scale = True,
                 normalize = True,
                 pi_rotate = True,
                 fixing_rotate = False,
                 angle_fix = 0):
        super(MoNuSegDataset).__init__()

        ## Transform
        self.mean = mean
        self.std = std 
        self.size_img = size_img
        self.size_crop = size_crop
        self.scale_factor = scale_factor
        self.p = p
        self.p_rotate = p_rotate
        self.rotate = rotate
        self.scale = scale
        self.normalize = normalize # Use un-normalize image for plotting
        self.pi_rotate = pi_rotate # Use only rotation 90,180,270 rotations
        self.fixing_rotate = fixing_rotate # For test time, allow to eval a model on the dataset with a certain angle
        self.angle_fix = angle_fix # The angle used for the fixing rotation

        if fixing_rotate: 
            self.rotate = False

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
        # Resize     
        if self.train and self.scale:
            min_size = int(self.size_img[0]*self.scale_factor[0])
            max_size = int(self.size_img[0]*self.scale_factor[1])
            if  min_size < self.size_crop[0]: 
                size = random.randint(self.size_crop[0],max_size)
            else:
                size = random.randint(min_size,max_size)
            resize = T.Resize((size,size))
        else:
            resize = T.Resize(self.size_img)
        image = resize(image)
        mask = resize(mask)

        if self.train : 
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=self.size_crop)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > self.p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            if self.rotate:
                if random.random() > self.p_rotate:
                    if self.pi_rotate:
                        angle = int(np.random.choice([90,180,270],1,replace=True)) #Only pi/2 rotation
                        image = TF.rotate(image,angle=angle)
                        mask = TF.rotate(mask,angle=angle)
                    else:
                        if random.random() > 0.5:
                            angle = np.random.randint(0,30)
                            image = TF.rotate(image,angle=angle)
                            mask = TF.rotate(mask,angle=angle)
                        else:
                            angle = np.random.randint(330,360)
                            image = TF.rotate(image,angle=angle)
                            mask = TF.rotate(mask,angle=angle)

        # Apply a fixed rotation for test time:
        if self.fixing_rotate:
            image = TF.rotate(image,angle=self.angle_fix)
            mask = TF.rotate(mask,angle=self.angle_fix)
        

        # Transform to tensor
        image = TF.to_tensor(image)
        if self.normalize:
            image = TF.normalize(image,self.mean,self.std)
        mask = to_tensor_target_lc(mask)
        return image, mask


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_img,self.file_names[index]+'.jpg')).convert('RGB')
        target = Image.open(os.path.join(self.root_img,self.file_names[index]+'_m.png'))
        img, target = self.my_transform(img, target)
        return img, target

    def __len__(self):
        return len(self.file_names)