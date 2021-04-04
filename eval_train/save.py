import os 
import random 
import numpy as np 
import torch 
from string import digits
def create_save_directory(path_save):
    """
        This function returned the path to a unique folder for each model running. 
    """
    files = os.listdir(path_save)
    n = [int(i) for i in files if i[0] in digits]
    if len(n)==0:
        d=0
    else:
        d = max(n)+1
    new_path = os.path.join(path_save,str(d)) #Return new path to save during training
    os.mkdir(new_path)
    return new_path

def save_curves(path,**kwargs):
    """
        path : path to save all the curves
        **kwargs : must be name_of_the_list = list
    """
    
    for name,l in kwargs.items():
        curve_name = os.path.join(path,str(name)+'.npy')
        if os.path.exists(curve_name): # If file exist save with an other name
            np.save(curve_name+str(random.randint(0,100)),np.array(l))
        else:
            np.save(curve_name,np.array(l))

def save_hparams(args,path):
    """
        Save hyperparameters of a run in a hparam.txt file
    """
    hparam = 'hparam.txt'
    fi = os.path.join(path,hparam)
    with open(fi,'w') as f:
        print(args,file=f)
    print('Hyper parameters succesfully saved in',fi)

def save_eval_angle(d_iou,save_dir):
    """
        Save the evaluation of IoU with different input angle rotation image in a file
    """
    angle = 'eval_all_angle.txt'
    fi = os.path.join(save_dir,angle)
    with open(fi,'a') as f:
        for k in d_iou.keys():
            print('Scores for datasets rotate by',k,'degrees:',file=f)
            print('   mIoU',d_iou[k]['mIoU'],'Accuracy',d_iou[k]['Accuracy'],'CE Loss',d_iou[k]['CE Loss'],file=f)
    print('Evaluation with different input rotation angle succesfully saved in',fi)

def save_model(model,save_all_ep,save_best,save_folder,model_name,ep=None,iou=None,iou_test=None):
    if save_all_ep:
        if ep is None:
            raise Exception('Saving all epochs required to have the epoch iteration.')
        save_model = model_name+'_ep'+str(ep)+'.pt'
        save = os.path.join(save_folder,save_model)
        torch.save(model,save)
    if save_best:
        if iou is None or iou_test is None:
            raise Exception('Saving best model required to pass the current IoU and the list of IoU in argument.')
        if len(iou_test)<=1:
            save_model = model_name+'.pt'
            save = os.path.join(save_folder,save_model)
            torch.save(model,save)
        else:
            if iou > max(iou_test[:len(iou_test)-1]):
                print('New saving, better IoU found')
                save_model = model_name+'.pt'
                save = os.path.join(save_folder,save_model)
                torch.save(model,save)
    else:
        save_model = model_name+'.pt'
        save = os.path.join(save_folder,save_model)
        torch.save(model,save)