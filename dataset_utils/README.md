# Utils files for the MoNuSeg dataset 

This is an update in Python3 of the existing utils code available on the dataset challenge website : https://monuseg.grand-challenge.org/
The previous code provided for the Challenge contains matlab, lua and caffe files, so i couldn't use it for my full python and pytorch setup. 
It's also required to download by your own the MoNuSeg dataset following this link : https://monuseg.grand-challenge.org/Data/ Training and Testing Data are available. 


## xml_to_mask.py

### This is used to create ternary maps from annotations and original image.



You need to change the following parameters in xml_to_mask:
- root_img : Root of the MoNuSeg images of size 1000x1000, 30 in train and 14 in test  (tif format). 
- root_ann : Root of the MoNuSeg annotations (xml format)
- binary_root : Path to be created to the save directory of the binary masks generated from the XML annotations.
- color_root : Path to be created to the save directory of the colored masks generated from the XML annotations.

## split.py

Once the xml files is converted to .png files and saved in the binary_root or color_root, the split.py file will generate a single folder 'Outputs'. In this folder, there are 21660 51x51 size patches saved as 'name'.jpg with its corresponding mask name_m.png format. It also write the name of all the generated patch in a list.txt file, it is more simple to use for the Pytorch dataset

## monuseg_dataset.py 

Contain the pytorch dataset loader for the MoNuSeg dataset. It's required to run the previous files (split.py and xml_to_mask.py) before you can use the Pytorch MoNuSeg dataset.

## my_transforms.py

Utils functions to apply transformation on image and ground truth mask input. 