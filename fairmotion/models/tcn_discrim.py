import torch.nn as nn
import numpy as np

'''
Implementation of the Discriminator Block for the TCN GAN architecture.
'''
class DilatedConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class TCNDiscriminator(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,linear_channels = [512,128,64,1],n_layers=5,int_length=20,activation='Sigmoid'):
        '''
        activation: Denotes the activation applied at the terminal layer. 
                    sigmoid (default), relu (applicable for LSGANS)
        
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        ni,no = in_channels,out_channels
        
        for idx in range(1,n_layers+1):
            self.layers.append(DilatedConv(ni,no,kernel_size,dilation=2**idx))
            ni = no
        ni = int_length*out_channels
        self.layers.append(nn.Flatten())
        for channel in range(len(linear_channels)-1):
            self.layers.append(nn.Linear(ni,linear_channels[channel]))
            self.layers.append(nn.ReLU())
            ni = linear_channels[channel]
        self.layers.append(nn.Linear(linear_channels[-2],linear_channels[-1]))
        if activation == 'Sigmoid':
            self.layers.append(nn.Sigmoid())
            
    def init_weights(self):
        for params in self.parameters():
            if params.data.ndimension()>=2:
                nn.init.xavier_normal_(params)
            else:
                nn.init.normal_(params,0,0.01)
    def forward(self,x):
        x = x.transpose(1,2)
        return nn.Sequential(*self.layers)(x)
    

        