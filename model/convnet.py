import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data



class CNN3(torch.nn.Module): # Inspire of the CNN3 archi of the MoNuSeg challenge paper
    def __init__(self):
        super(CNN3,self).__init__()

        self.conv = nn.Sequential(
            # 51 x 51
            nn.Conv2d(in_channels=3,out_channels=25,kernel_size=(4,4)), # [48,48]
            nn.ReLU(inplace=True),
            nn.Dropout2D(p=0.1),
            nn.MaxPool2d(kernel_size=(2,2)), # [24,24]
            nn.Conv2d(in_channels=25,out_channels=50,kernel_size=(5,5)), # [20,20]
            nn.ReLU(inplace=True),
            nn.Dropout2D(p=0.2),
            #nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=50,out_channels=80,kernel_size=(4,4)), # [17,17]
            nn.ReLU(inplace=True),
            nn.Dropout2D(p=0.25),
            #nn.MaxPool2d(kernel_size=(2,2))
        )        
        self.conv_classif = nn.Sequential(
            nn.Conv2d(in_channels=80,out_channels=80,kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=80,out_channels=3,kernel_size=(1,1))
        )

        self.upsampling = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=3)#, [17,17] --> [51,51] original size
        )


    def forward(self,x):
        # [51,51]
        x = self.conv(x)
        # [17,17]
        x = self.conv_classif(x)
        # [17,17]
        x = self.upsampling(x)
        # [51,51]

        return x # [51,51]