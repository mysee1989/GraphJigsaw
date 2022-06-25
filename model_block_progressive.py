import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
# from models.resnet import resnet50
from models.my_resnet import resnet50
from models.gat import GraphAttentionLayer
import pdb

# graph jigsaw puzzle network
class GJPN(nn.Module):
    def __init__(self,backbone,feature_size,classes_num,block_num, phase = 'train'):
        super(GJPN,self).__init__()

        self.feature = backbone
        self.num_ftrs = 2048 * 1 * 1
        self.classes_num = classes_num
        self.block_num = block_num
        self.phase = phase

      
        self.roi_pooling = nn.AdaptiveAvgPool2d((block_num,block_num))
        self.max1 = nn.MaxPool2d(kernel_size=28,stride=28)
        self.max2 = nn.MaxPool2d(kernel_size=14,stride=14)
        self.max3 = nn.MaxPool2d(kernel_size=7,stride=7)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num)
        )

        # if you want use GAT, call GraphAttentionLayer(in_features, out_features)
        
        # ************ For resnet backbone ************#  
        #stage 1        
        self.gc1 = GraphConvolution(self.num_ftrs//8,self.num_ftrs//16)
        self.gc2 = GraphConvolution(self.num_ftrs//16,self.num_ftrs//8)

        #stage 2        
        self.gc1_1 = GraphConvolution(self.num_ftrs//4,self.num_ftrs//8)
        self.gc2_1 = GraphConvolution(self.num_ftrs//8,self.num_ftrs//4)

        #stage 3
        self.gc1_2 = GraphConvolution(self.num_ftrs//2,self.num_ftrs//4)
        self.gc2_2 = GraphConvolution(self.num_ftrs//4,self.num_ftrs//2)

        #stage 4
        self.gc1_3 = GraphConvolution(self.num_ftrs,self.num_ftrs//2)
        self.gc2_3 = GraphConvolution(self.num_ftrs//2,self.num_ftrs)

        # ************ For densenet169 backbone ************#  
        # #stage 1        
        # self.gc1 = GraphConvolution(64,128)
        # self.gc2 = GraphConvolution(128,256)

        # #stage 2        
        # self.gc1_1 = GraphConvolution(128,256)
        # self.gc2_1 = GraphConvolution(256,512)

        # #stage 3
        # self.gc1_2 = GraphConvolution(256,512)
        # self.gc2_2 = GraphConvolution(512,1280)

        # #stage 4
        # self.gc1_3 = GraphConvolution(640,1024)
        # self.gc2_3 = GraphConvolution(1024,1664)

    def forward(self,x,epoch,batch_id,after_epoch,phase='train'):
        xl1_b1,xl1_b3,xl2_b1,xl2_b4,xl3_b1,xl3_b6,xl4_b1,xl4_b3 = self.feature(x)
        
        if phase == 'test':
            output =self.max3(xl4_b3)
            output = output.view(output.size(0),-1)
            output = self.classifier[0](output)
            output = self.classifier[1](output)
            return output
        
        #classifier
        output = self.max3(xl4_b3)
        output = output.view(output.size(0),-1)
        output = self.classifier(output)
        if epoch<=after_epoch:    
            return output

        #gcn inference and recontrust
        #stage 1
        if batch_id%4 ==0:
            xl1_b1 = self.roi_pooling(xl1_b1)
            xl1_b1 = xl1_b1.reshape((xl1_b1.size(0),xl1_b1.size(1),xl1_b1.size(2)*xl1_b1.size(3)))
            xl1_b1 = xl1_b1[:,:,torch.randperm(xl1_b1.size(2))]
            xl1_b1 = xl1_b1.permute(0,2,1)

            adj = adjacent_matrix_generator(self.block_num,xl1_b1.size(0)).cuda()
            xl1_b1 = F.relu(self.gc1(xl1_b1,adj))
            recon_feature = self.gc2(xl1_b1,adj)

            xl1_b3 = self.roi_pooling(xl1_b3)
            xl1_b3 = xl1_b3.reshape((xl1_b3.size(0),xl1_b3.size(1),xl1_b3.size(2)*xl1_b3.size(3)))
            feature = xl1_b3.permute(0,2,1)

        # # stage 2
        if batch_id%4 == 1:
            xl2_b1 = self.roi_pooling(xl2_b1)
            xl2_b1 = xl2_b1.reshape((xl2_b1.size(0),xl2_b1.size(1),xl2_b1.size(2)*xl2_b1.size(3)))
            xl2_b1 = xl2_b1[:,:,torch.randperm(xl2_b1.size(2))]
            xl2_b1 = xl2_b1.permute(0,2,1)

            adj = adjacent_matrix_generator(self.block_num,xl2_b1.size(0)).cuda()
            xl2_b1 = F.relu(self.gc1_1(xl2_b1,adj))
            recon_feature = self.gc2_1(xl2_b1,adj)

            xl2_b4 = self.roi_pooling(xl2_b4)
            xl2_b4 = xl2_b4.reshape((xl2_b4.size(0),xl2_b4.size(1),xl2_b4.size(2)*xl2_b4.size(3)))
            feature = xl2_b4.permute(0,2,1)

        # stage 3
        elif batch_id%4 == 2:
            xl3_b1 = self.roi_pooling(xl3_b1)
            xl3_b1 = xl3_b1.reshape((xl3_b1.size(0),xl3_b1.size(1),xl3_b1.size(2)*xl3_b1.size(3)))
            xl3_b1 = xl3_b1[:,:,torch.randperm(xl3_b1.size(2))]
            xl3_b1 = xl3_b1.permute(0,2,1)

            adj = adjacent_matrix_generator(self.block_num,xl3_b1.size(0)).cuda()
            xl3_b1 = F.relu(self.gc1_2(xl3_b1,adj))
            recon_feature = self.gc2_2(xl3_b1,adj)

            xl3_b6 = self.roi_pooling(xl3_b6)
            xl3_b6 = xl3_b6.reshape((xl3_b6.size(0),xl3_b6.size(1),xl3_b6.size(2)*xl3_b6.size(3)))
            feature = xl3_b6.permute(0,2,1)

        # stage 4
        else:
            xl4_b1 = self.roi_pooling(xl4_b1)
            xl4_b1 = xl4_b1.reshape((xl4_b1.size(0),xl4_b1.size(1),xl4_b1.size(2)*xl4_b1.size(3)))
            xl4_b1 = xl4_b1[:,:,torch.randperm(xl4_b1.size(2))]
            xl4_b1 = xl4_b1.permute(0,2,1)

            adj = adjacent_matrix_generator(self.block_num,xl4_b1.size(0)).cuda()
            xl4_b1 = F.relu(self.gc1_3(xl4_b1,adj))
            recon_feature = self.gc2_3(xl4_b1,adj)

            xl4_b3 = self.roi_pooling(xl4_b3)
            xl4_b3 = xl4_b3.reshape((xl4_b3.size(0),xl4_b3.size(1),xl4_b3.size(2)*xl4_b3.size(3)))
            feature = xl4_b3.permute(0,2,1)


        return output,recon_feature,feature


def rearrange(x,patch_size,patch_num):
    x_new = torch.empty(x.size(0),x.size(1)*patch_size*patch_size,patch_num,patch_num).cuda()
    for i in range(patch_num):
        for j in range(patch_num):
            tmp = x[:,:,patch_size*i:patch_size*i+patch_size,patch_size*j:patch_size*j+patch_size].clone()
            tmp = tmp.reshape(tmp.size(0),tmp.size(1)*tmp.size(2)*tmp.size(3))
            x_new[:,:,i,j] = tmp
    return x_new.cuda()

class GCN_Block(nn.Module):
    def __init__(self,input_channel):
        super(GCN_Block,self).__init__()
        self.input_channel = input_channel

        self.gc1 = GraphConvolution(self.input_channel,self.input_channel//2)
        self.gc2 = GraphConvolution(self.input_channel//2,self.input_channel)
    
    def forwad(self,x,adj):
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x,0.5)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x,0.5)
        x = self.gc3(x,adj)

        return x



def adj_normalize(adj):
    D = torch.pow(adj.sum(1).float(),-0.5)
    D = torch.diag(D)
    adj_nor = torch.matmul(torch.matmul(adj,D).t(),D)
    return adj_nor

def adjacent_matrix_generator(block_num,batch_size):
    adj = np.zeros((batch_size,block_num*block_num,block_num*block_num))
    for i in range(block_num*block_num):
        for j in range(block_num*block_num):
            if i == j:
                adj[:,i,j] = 1
            if (i+1)%block_num ==0:
                if j == i-1 or j == i-block_num or j == i +block_num:
                    adj[:,i,j] = 1
            elif i%block_num == 0:
                if j == i+1 or j == i-block_num or j == i +block_num:
                    adj[:,i,j] = 1
            else:
                if j == i+1 or j == i-1 or j == i-block_num or j == i +block_num:
                    adj[:,i,j] = 1
    adj = torch.from_numpy(adj).float()
    for i in range(batch_size):
        adj[i] = adj_normalize(adj[i])
    return adj

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class PositionalEncoding_learnable(nn.Module):
    def __init__(self,n_position,channel):
        super(PositionalEncoding_learnable,self).__init__()

        self.position_embedding = Parameter(torch.empty(size=(1,n_position,channel)),requires_grad=True)
        nn.init.xavier_uniform_(self.position_embedding.data, gain=1.414)

    def forward(self,x):
        return x + self.position_embedding

class PositionalEncoding_sin(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding_sin, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()



