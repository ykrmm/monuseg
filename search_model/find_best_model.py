import os 
from argparse import ArgumentParser
import numpy as np
import re 
import torch
import sys
sys.path.insert(1, '../utils')
import argparse

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

def get_max_file(file='iou_test.npy'):
    maxx = 0 
    argmax = 0
    try:
        iou = np.load(file)
        maxx = max(iou)
        argmax = np.argmax(iou)
    except:
        print("No iou found in:",file)
    return maxx,argmax


def find_best_model_(iou_file='iou_test.npy',**kwargs):
    best = 0
    list_folder = [f[0] for f in os.walk(kwargs['folder'])][1:] # The first one is the current directory which we're not interested
    for f in list_folder:
        param_file = os.path.join(f,'hparam.txt')
        with open(param_file,'r') as fi:
            param = fi.read()
        match = [] # Parameters in the file
        launch_search = True
        print('Search in',f)
        for k in kwargs.items():
            if k[0]!='folder' and k[1] is not None:
                t = type(k[1])
                search = k[0]+'='
                s = re.search('(,\s)('+search+')(.*?)(?=,)',param,re.IGNORECASE) # search the argument in the param file
                #print("K",k)
                try:
                    c = s.group(3)
                except:
                    print("Didn't find argument",k,"in the hparam file")
                    launch_search = True
                #   
                #print("C",c) 
                #print("T", t) 
                match.append(str(c)==str(k[1]))
                if not str(c)==str(k[1]):
                    launch_search = False
        if launch_search:
            print('Match in',f)
            maxx,argmax = get_max_file(os.path.join(f,iou_file))
            if maxx > best:
                best_dir = f
                best = maxx
    return best_dir,best


def load_best_model(iou_file='iou_test.npy',**kwargs):
    best = 0
    argmax_bf = -1
    list_folder = [f[0] for f in os.walk(kwargs['save_dir'])][1:] # The first one is the current directory which we're not interested
    print(list_folder)
    for f in list_folder:
        param_file = os.path.join(f,'hparam.txt')
        with open(param_file,'r') as fi:
            param = fi.read()
        match = [] # Parameters in the file
        launch_search = True
        for k in kwargs.items():
            if k[0]!='save_dir' and k[0]!='model_name' and k[1] is not None:
                t = type(k[1])
                search = k[0]+'='
                s = re.search('('+search+')(.*?)(?=,)',param,re.IGNORECASE) # search the argument in the param file
                c = s.group(2)      
                if t==type(False):
                    print("C",c)
                    c = str2bool(c)
                    print(c)
                else:
                    c = t(c)
                match.append(c==k[1])
                if not c==k[1]:
                    #print('c',c)
                    #print('k',k)
                    #print('FALSE')
                    launch_search = False

        if launch_search:
            print('search in',f)
            maxx,argmax = get_max_file(os.path.join(f,iou_file))
            if maxx > best:
                best_dir = f
                best = maxx
                argmax_bf = argmax
    try:
        model_name = kwargs['model_name']
        try:
            load_pt = model_name+'_ep'+str(argmax_bf)+'.pt'
            model = torch.load(os.path.join(best_dir,load_pt))
            print('Success to load',load_pt)
        except:
            print('In folder',best_dir,'not all epoch was saved')
            load_pt = model_name+'.pt'
            model = torch.load(os.path.join(best_dir,load_pt))
            print('Success to load',load_pt)
    except:
        raise Exception("No argument 'model_name' in params")
    return model,best_dir


## À FAIRE 
# Plot les courbes des bests models 
# 
def main():

    parser = ArgumentParser()
    parser.add_argument('--folder',default='/share/homes/karmimy/equiv/save_model/fully_supervised',type=str)
    parser.add_argument('--split', type=str2bool, help="Split the dataset")
    parser.add_argument('--split_ratio', type=float, help="Amount of data we used for training")
    parser.add_argument('--batch_size', type=float, help="Batch size") 
    parser.add_argument('--rotate', type=str2bool, help="Rotate")   
    args = parser.parse_args()

    #print(**vars(args))
    print(args)
    best_dir,best = find_best_model_(**vars(args))

    print("The best model is in",best_dir,"with an iou of",best)

if __name__ == '__main__':
    main() 