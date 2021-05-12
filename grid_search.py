#!/usr/bin/env python3

from eval_train.utils import clean_prediction
import glob
import os
from os.path import join
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch 
from dataset_utils import MoNuSegDataset
from model import CNN3
from PIL import Image 
from eval_train import compute_AJI
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2


ROOT = '/share/DEEPLEARNING/datasets/monuseg/'

MODEL_DIR = '/share/homes/karmimy/equiv/save_model/fully_supervised_monuseg/48'
MODEL_NAME = 'fully_supervised_monuseg.pt'
gpu = 1

test_dataset = MoNuSegDataset(ROOT,image_set='test',load_entire_image=True,binary=False)
dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=4,pin_memory=True,\
            batch_size=1)
device =  torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
print('DEVICE',device)
model = torch.load(join(MODEL_DIR,MODEL_NAME))
model.to(device)



best_aji = 0
best_param = {'dist_factor':0,'clean_prediction':None,'it_bck':0,'it_opening':0}

d_factor = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]
clean_p = [False]
it_b = [0,1,2,3]
it_o = [0,1,2,3]
iter = len(d_factor)* len(clean_p) *len(it_b)* len(it_o)
cpt = 0
for d in d_factor:
    for c in clean_p:
        for ib in it_b:
            for io in it_o:
                aji,aji_mean = compute_AJI(model,dataloader_val,device,dist_factor=d,threshold=54,clean_prediction=c,it_bg=ib,it_opening=io) 
                if aji_mean> best_aji:
                    best_param['dist_factor'] = d
                    best_param['clean_prediction'] = c
                    best_param['it_bck'] = ib 
                    best_param['it_opening'] = io
                    best_aji = aji_mean
                print('ITER:',cpt,'/',iter)
                cpt +=1

print('GRID SEARCH DONE, BEST PARAMS FOR WATERSHED ARE',best_param,'FOR AN AJI OF:',best_aji)

"""
GRID SEARCH DONE, BEST PARAMS FOR WATERSHED ARE {'dist_factor': 0.3, 'clean_prediction': False, 'it_bck': 0, 'it_opening': 0} FOR AN AJI OF: 0.6097596976260251
"""