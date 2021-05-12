import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(image, size, fill=0):
    print(image.size)
    min_size = min(image.size)
    
    if min_size < size:
        ow, oh = image.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        image = F.pad(image, (0, 0, padw, padh), fill=fill)
    return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            if type(t)==type([]):
                for tt in t:
                    image,target = tt(image,target)
            else:
                image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        if self.min_size == self.max_size:
            return image,target
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target

class RandomRotate(object):
    def __init__(self,angle_max,p_rotate,expand):
        self.angle = angle_max
        self.p_rotate = p_rotate
        self.expand = expand

    def __call__(self,image,mask):
        if random.random() > self.p_rotate:
            if random.random() > 0.5:
                angle = np.random.randint(0,self.angle)
                mask = torch.unsqueeze(mask,0)
                image = F.rotate(image,angle=angle,expand=self.expand)
                mask = F.rotate(mask,angle=angle,expand=self.expand)
                mask = mask.squeeze()
            else:
                angle = np.random.randint(360-self.angle,360)
                mask = torch.unsqueeze(mask,0)
                image = F.rotate(image,angle=angle,expand=self.expand)
                mask = F.rotate(mask,angle=angle,expand=self.expand)
                mask = mask.squeeze()
        return image,mask

class RandomPiRotate(object):
    def __init__(self,angle,p_rotate):
        self.angle = angle
        self.p_rotate = p_rotate

    def __call__(self,image,mask):
        if random.random() > self.p_rotate:
            angle = int(np.random.choice([90,180,270],1,replace=True)) #Only pi/2 rotation
            image = F.rotate(image,angle=angle)
            mask = F.rotate(mask,angle=angle)

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        #image = pad_if_smaller(image, self.size)
        #target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __init__(self,border=False):
        self.border = border

    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = np.array(target)
        if not self.border:
            target[target==255] = 0 # border = background 
        return image, torch.LongTensor(target)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0,saturation=0,hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        if self.brightness is not None:
            image = F.adjust_brightness(image, self.brightness)
        if self.contrast is not None:
            image = F.adjust_contrast(image, self.contrast)
        if self.saturation is not None:
            image = F.adjust_saturation(image, self.saturation)
        if self.hue is not None:
            image = F.adjust_hue(image, self.hue)
        return image, target