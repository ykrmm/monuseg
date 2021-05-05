import numpy as np 
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max







class AJI_Metrics(object):
    def __init__(self) -> None:
        self.aji = [] # aji metrics 

    def add_prediction(self,pred,gt):
        intersection = 0
        union = 0
        pred = pred.squeeze()
        gt = gt.squeeze()
        pred_np = pred.argmax(dim=0).detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        # Watershed prediction
        distance = ndi.distance_transform_edt(pred_np)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=pred_np)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels_pred = watershed(-distance, markers, mask=pred_np)

        # Watershed ground truth
        distance = ndi.distance_transform_edt(gt)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=gt)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels_gt = watershed(-distance, markers, mask=gt) # Probleme --> Watershed devrait être utiliser que pour les prédictions pas pour la GT 
        

        markers_pred = np.zeros(np.shape(np.unique(labels_pred)))  # 0 if unmarked 1 if marked 
        list_g = np.unique(labels_gt)
        markers_pred[0] = 1 # Mark background as used
        
        for g in list_g[1:]: # Don't take the label 0 (background)
            mask_g = np.where(labels_gt!=g,0,labels_gt)
            res = np.where(mask_g == g ) # Coord where the nuclei g is in the ground truth mask 
            overlap = []
            for i in range(len(res[0])):
                coord = (res[0][i],res[1][i])
                overlap.append(labels_pred[coord])

            overlap = np.unique(overlap) # Get overlap nuclei in pred masks 

            iou = []
            l_inter = []
            l_union = []
            for i,s in enumerate(overlap):
                if markers_pred[s] ==0:
                    mask_s = np.where(labels_pred!=s,0,labels_pred)
                    mask_s = np.where(mask_s==s,g,mask_s)
                    inter = np.logical_and(mask_g, mask_s)
                    u = np.logical_or(mask_g, mask_s)
                    inter = np.sum(inter)
                    u = np.sum(u)
                    l_inter.append(inter)
                    l_union.append(u)
                    iou_score = inter / u
                    iou.append(iou_score)
                else:
                    inter = 0
                    u = 0
                    l_inter.append(inter)
                    l_union.append(u)
                    iou_score = 0
                    iou.append(iou_score)
            try:
                best = np.argmax(iou)
                best_s = overlap[best] 
                intersection += l_inter[best]
                union += l_union[best]
                markers_pred[best_s] = 1 # Mark the best nuclei used
            except:
                occurence = np.count_nonzero(labels_gt ==g) # If overlap is empty, then its a false positive 
                union+= occurence
                print('Overlap nuclei list of nuclei',g,'is empty')

        # Add to union every pred segmented nuclei unused
        for i,m in enumerate(markers_pred):
            if m ==0 : 
                occurence = np.count_nonzero(labels_pred ==i)
                union+= occurence

        aji = float(intersection)/float(union)
        self.aji.append(aji)
        return aji

    def get_aji(self):
        "When all aji are computed on the dataset, This function return the average aji"

        return np.array(self.aji).mean()  # Average of AJI
    

    def get_all_aji(self):
        return self.aji