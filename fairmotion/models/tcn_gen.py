import torch.nn as nn
import torch
import numpy as np
class TrailingPadding(nn.Module):
    '''
    Removes the padding from the input tensor after applying 1D convolution to avoid
    dimensional inconsistencies

    '''
    def __init__(self,padding):
        super().__init__()
        self.padding = padding
    def forward(self,x):return x[:,:,:-self.padding]

class CausalConv1d(nn.Module):
    '''
    Implementation of 1D Dilated-Causal Convolution layer for the TCN architecture.
    '''

    def __init__(self,in_channels,out_channels,kernel_size,dilation=1,dropout=0.2):
        super().__init__()
        pad          = (kernel_size-1)*dilation
        self.conv    = nn.Conv1d(in_channels,out_channels,kernel_size,padding=pad,dilation=dilation)
        self.relu    = nn.ReLU()
        self.bn      = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.trail   = TrailingPadding(pad)
  
    def forward(self,x):
        return self.dropout(self.relu(self.bn(self.trail(self.conv(x)))))

class TCNLayer(nn.Module):
    '''
    Single TCN layer with provision to extend that to multiple layers
    '''
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2,n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        ni,no = n_inputs,n_outputs
        for _ in range(n_layers):
            self.layers.append(CausalConv1d(ni,no,kernel_size,dilation=dilation,dropout=dropout))
            ni = no
    def forward(self, x):
        out = nn.Sequential(*self.layers)(x)
        return out

class Attention(nn.Module):
    '''
    Implements a temporal attention block with a provision to increase the number of
    heads to two

    n_heads: 1
    activation: softmax (default), tanh
    '''
    def __init__(self,input_dims,attention_dims,n_heads=1,activation='softmax'):
        super().__init__()
        self.attention_dims = attention_dims
        self.n_heads = n_heads
        self.k1 = nn.Linear(input_dims, attention_dims)
        self.v1 = nn.Linear(input_dims, attention_dims)
        self.q1 = nn.Linear(input_dims, attention_dims)
        
        if n_heads == 2:
            self.k2 = nn.Linear(input_dims, attention_dims)
            self.v2 = nn.Linear(input_dims, attention_dims)
            self.q2 = nn.Linear(input_dims, attention_dims)
            self.attention_head_projection = nn.Linear(attention_dims*2,input_dims)
        else:
            self.attention_head_projection = nn.Linear(attention_dims,input_dims)

        if activation == 'Softmax':self.activation = nn.Softmax(dim=2)
        elif activation == 'tanh':self.activation  = nn.Tanh(dim=2)

    def forward(self,x):
        '''
        x: shape (B,H,T) where B is the Batch size, H is the feature size, and T is sequence length

        '''
        x = x.transpose(2,1)
        q1,v1,k1    = self.q1(x),self.v1(x),self.k1(x)
        qk1         = (q1@k1.permute((0,2,1)))/np.sqrt(self.attention_dims)
        attention_1    = self.softmax(qk1)@v1 
        if self.n_heads == 2:
            q2,v2,k2    = self.q2(x),self.v2(x),self.k2(x)
            qk2         = (q2@k2.permute((0,2,1)))/np.sqrt(self.attention_dims) 
            attention_2 =  self.softmax(qk2)@v2       
            multihead = torch.cat((attention_1,attention_2),dim=-1)
        else:multihead = attention_1
        multihead_concat = self.attention_head_projection(multihead)     
        return multihead_concat.transpose(2,1)
        
class ResidualTCN(nn.Module):
    '''
    Implements a Residual TCN Block with 5 blocks in each layer along with 
    an exponentially increasing dilation factor
    '''
    def __init__(self,n_inputs,n_outputs,kernel_size,n_blocks=5,attention=False,attention_dims=64):
        super().__init__()
        self.layers = nn.ModuleList()
        ni,no = n_inputs,n_outputs
        for block in range(n_blocks):
            self.layers.append(TCNLayer(ni,no,kernel_size,dilation = 2**block))
            if attention:self.layers.append(Attention(no,attention_dims))
            ni = no
        self.layers.append(TCNLayer(no,n_inputs,kernel_size,dilation=2**(n_blocks-1)))

    def forward(self,x):
        out = nn.Sequential(*self.layers)(x)
        return out+x

    
class TCN(nn.Module):

    '''
    Implementation of the entire TCN architecture as proposed with Xavier Normalization
    '''
    def __init__(self,n_inputs,n_outputs,kernel_size,n_blocks=5,linear=[64,32,32],target_length=24,frame_length=120,attention=False,attention_dims=64):
        super().__init__()
        self.layers = nn.ModuleList()
        ni,no = n_inputs,n_outputs
        for _ in range(n_blocks-1):
            self.layers.append(ResidualTCN(ni,no,kernel_size,attention=attention,attention_dims=attention_dims))
        
        ni = frame_length
        for i in range(len(linear)):
            self.layers.append(nn.Linear(ni,linear[i]))
            ni = linear[i]
        self.layers.append(nn.Linear(linear[-1],target_length))

    def init_weights(self):
        for params in self.parameters():
            if params.data.ndimension()>=2:
                nn.init.xavier_normal_(params)
            else:
                nn.init.normal_(params,0,0.01)
        
    def forward(self,x):
        '''
        Reshape the input tensor to match the 1D Convolution Requirements
        x: shape (B,H,T) where B is the Batch size, H is the feature size, and T is sequence length
        '''
        x   = x.transpose(2,1)
        out = nn.Sequential(*self.layers)(x)
        return out.transpose(1,2)