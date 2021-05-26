import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage.interpolation import rotate as scipy_rotate
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2

###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            EQUIVARIANCE UTILS FUNCTION
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

# rotate images
def rotate_image(image,angle,reshape=False):
    """
        Rotate a tensor with a certain angle.
        If true, expands the output image to make it large enough to hold the entire rotated image.
        Else it keeps the same size
        Depreciated...
    """
    #image = image.squeeze()
    if len(image.size())==3: # Case of a single image.
        axes = ((1,2))
    elif len(image.size())==4: # Case of a batch of images
        axes = ((2,3))
    else:
        print("Dimension of images must be 4 or 5.")
        return 
    im = scipy_rotate(image.numpy(),angle=angle,reshape=reshape,axes=axes)
    im_t = torch.FloatTensor(im)
    return (im_t,360-angle)

def rotate_pt(img,angle,reshape=False):
    """
        Rotate a tensor with a certain angle.
        If true, expands the output image to make it large enough to hold the entire rotated image.
        Else it keeps the same size
    """
    img = TF.rotate(img,angle=30,expand=reshape)

    return (img,360-angle)


def rotate_mask(mask,angle,reshape=False):
    """
        This function take a prediction from the model [batch_size,21,513,513] 
        and rotate, by an angle add as a parameters, the prediction.
        To make sure there is no error it is preferable to use new_angle returned by the function 'rotate_image'.
    """
    with torch.no_grad():
        if len(mask.size())==3: # Case of a single mask.
            axes = ((1,2))
        elif len(mask.size())==4: # Case of a batch of masks
            axes = ((2,3))
        else:
            print("Size must be 4 or 5.")
            return 
        m = scipy_rotate(mask.numpy(),angle=angle,reshape=reshape,axes=axes,mode='nearest')
        mask_t = torch.FloatTensor(m)
        return mask_t
    
def compute_transformations_batch(x,model,angle,reshape=False,\
                                  criterion=nn.KLDivLoss(reduction='batchmean'),Loss=None,rot_cpu=False,device='cpu',plot=False):
    """
       This function compute the equivariance loss with the rotation transformation for a batch of images. 
       It also give the accuracy between the output produce by the original image and the outpute produce by the 
       transforme image.
       criterion : KL divergence / L1 Loss / MSE Loss
       Loss : 'str' ; 'KL' or 'CE' or None
       rot_cpu : if True the rotation will process on CPU -> More time but less gpu usage
       plot = True for debug
       reshape = True to allow to grow the images during the rotation to not loose the border
    """
    x = x.to(device)
    if rot_cpu:
        rot_x,_= rotate_pt(x.detach().cpu(),angle=angle,reshape=reshape) 
    else:
        rot_x,_ = rotate_pt(x,angle=angle,reshape=reshape)
    logsoftmax = nn.LogSoftmax(dim=1) #LogSoftmax using instead of softmax then log.
    softmax = nn.Softmax(dim=1)
    try:
        pred_x = model(x.to(device))['out'] # a prediction of the original images.
        pred_rot = model(rot_x.to(device))['out'] # a prediction of the rotated images.
    except:
        pred_x = model(x.to(device))
        pred_rot = model(rot_x.to(device))

    if rot_cpu:    
        pred_rot_x,_ = rotate_pt(pred_x.detach().cpu(),angle,reshape=reshape) # CPU rotation 
    else:
        pred_rot_x,_ = rotate_pt(pred_x,angle,reshape=reshape) # Apply the rotation on the mask with the original input

    if Loss=='KL':
        loss = criterion(logsoftmax(pred_rot_x.cpu()),softmax(pred_rot.cpu())) #KL divergence between the two predictions
        loss = loss/ (pred_x.size()[2]*pred_x.size()[3]) # Divide by the number of pixel in the image. Essential for batchmean mode in KLDiv
    elif Loss == 'CE':
        loss = criterion(pred_rot.cpu(),pred_rot_x.argmax(dim=1).detach().cpu()) # Use the prediction on the original image as GTruth.  
    else:
        loss = criterion(pred_rot_x.cpu(),pred_rot.cpu()) # For loss L1, MSE…    
    acc = float(torch.sum(pred_rot_x.argmax(dim=1).cpu()==pred_rot.argmax(dim=1).cpu())/(pred_rot_x.size()[2]**2))
    # compare the pred on the original images and the pred on the rotated images put back in place
    
        
        
    return loss,acc  


def eval_accuracy_equiv(model,val_loader,criterion=nn.KLDivLoss(reduction='batchmean'),\
                        nclass=21,device='cpu',Loss='KL',plot=True,angle_max=30,random_angle=False):
    """
        Function to compute the accuracy between the mask where the input had a geometric transformation 
        and the mask geometric transformed with the original input.
        random_angle -> boolean : If true a Random angle between 0 and angle_max is used for the evaluation.
        angle_max -> float : The max angle for rotate the input. 
        plot -> boolean : True plot the two masks side by side.
        Loss -> type of loss used : 'KL', 'CE' or None. 
        
    """    
    loss_test = []
    pixel_accuracy = []
    model.eval()
    with torch.no_grad():
        for i,(x,mask) in enumerate(val_loader):
            if random_angle:
                angle = np.random.randint(0,angle_max)
            else:
                angle = angle_max

            loss_equiv,acc = compute_transformations_batch(x,model,angle,reshape=False,\
                                                     criterion=criterion,Loss = Loss,\
                                                       device=device)
            loss_test.append(loss_equiv)
            pixel_accuracy.append(acc)

    m_pix_acc = np.array(pixel_accuracy).mean()
    m_loss_equiv = np.array(loss_test).mean()
    print("Mean Pixel Accuracy between masks :",m_pix_acc,"Loss Validation :",m_loss_equiv)
    return m_pix_acc, m_loss_equiv

###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                               WATERSHED POST PROCESSING
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|


def watershed_prediction(pred:torch.Tensor,clean_pred=False,threshold=54,dist_factor=0.3,compactness=0,it_opening=2,it_bg=3):
    """
        Clean prediction -> float If true it'll remove of the prediction the nuclei that occure less than threshold
        threshold -> int: Threshold for cleaning predictions
        dist_factor -> float : Distance factor for the distance map -> It controll the number of nuclei after the watershed process
    """
    pred = pred.squeeze()
    pred_np = pred.argmax(dim=0).detach().cpu().numpy()
    # noise removal 
    pred_np =  np.uint8(pred_np)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(pred_np,cv2.MORPH_OPEN,kernel, iterations = it_opening)

    # sure background area
    sure_bg = cv2.dilate(pred_np,kernel,iterations=it_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,dist_factor*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    labels_pred = watershed(-dist_transform, markers, mask=pred_np,compactness=compactness)
    # Add one to all labels so that sure background is not 0, but 1

    if clean_pred:
        labels_pred = clean_prediction(labels_pred,threshold)

    return labels_pred


def clean_prediction(labels_pred,threshold):
    """
        Function that supress small nuclei predictions under the threshold
    """
    count_pred = np.unique(labels_pred,return_counts=True)
    for i,c in zip(count_pred[0],count_pred[1]):
        if c <threshold:
            labels_pred = np.where(labels_pred==i,0,labels_pred)

    
    return labels_pred
