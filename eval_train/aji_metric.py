import numpy as np 
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max



class AJI_Metrics(object):
    def __init__(self,size_img=(1000,1000)) -> None:
        self.aji = 0 # aji metrics 
        self.cpt = 0 # count how many images are added to the class  

    def add_prediction(pred,gt):
        # Now we want to separate the two objects in image
        # Generate the markers as local maxima of the distance to the background
        intersection = 0
        union = 0
        pred = pred.squeeze()
        gt = gt.squeeze()
        pred_np = pred.argmax(dim=0).detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        distance = ndi.distance_transform_edt(pred_np)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=pred_np)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels_pred = watershed(-distance, markers, mask=pred_np)

        

        distance = ndi.distance_transform_edt(gt)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=gt)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels_gt = watershed(-distance, markers, mask=gt)

        markers = np.zeros(np.shape(labels_pred)) # 0 if unmarked 1 if marked 


        
        for g in labels_gt:
            