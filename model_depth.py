from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from slice import bilateral_slice
import os

from hediffmodel import DiffusionNet

class L2LOSS(nn.Module):
    def forward(self, x,y):
        return torch.mean((x-y)**2)


class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

        if use_bias and not batch_norm:
            self.conv.bias.data.fill_(0.00)
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None  
        
        if not batch_norm:
            self.fc.bias.data.fill_(0.00)
        torch.nn.init.kaiming_uniform_(self.fc.weight)
        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        bilateral_grid = bilateral_grid.permute(0,3,4,2,1)
        guidemap = guidemap.squeeze(1)
        coeefs = bilateral_slice(bilateral_grid, guidemap).permute(0,3,1,2)
        return coeefs

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 9:10, :, :]
        G = torch.sum(full_res_input * coeff[:, 3:6, :, :], dim=1, keepdim=True) + coeff[:, 10:11, :, :]
        B = torch.sum(full_res_input * coeff[:, 6:9, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, params['guide_complexity'], kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(params['guide_complexity'], 1, kernel_size=1, padding=0, activation= nn.Sigmoid) #nn.Tanh nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))

class Coeffs(nn.Module):

    def __init__(self, nin=4, nout=3, params=None,ddpm_ckpt=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin 
        self.nout = nout
        
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']
        bn = params['batch_norm']
        nsize = params['net_input_size']

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm*(2**i)*lb
    
        self.splat_addn_layer = nn.Conv2d(256, prev_ch, kernel_size=2, stride=2) # nn.Conv2d(640, prev_ch, kernel_size=1)

        self.splat_cct_layer = nn.Conv2d(128, prev_ch, kernel_size=1, padding=0)


        # global features
        n_layers_global = int(np.log2(sb/4))
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm*8*lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm*8*lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize/2**n_total)**2
        self.global_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8*cm*lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8*cm*lb, 8*cm*lb, 3, activation=None, use_bias=False))
        
        # predicton
        self.conv_out = ConvBlock(8*cm*lb, lb*nout*nin, 1, padding=0, activation=None)#,batch_norm=True)

        #depth params predictor 
        self.depth_param_predictor = nn.Sequential(ConvBlock(8*cm*lb, 64, kernel_size=3, stride=2),
                                                    ConvBlock(8*cm*lb, 64, kernel_size=3, stride=2),
                                                    nn.AvgPool2d(stride=4, kernel_size=4))

        self.depth_param_predictor_fc = nn.Linear(64, 4)

        self.diffmodel = DiffusionNet(dim=32, channels=3, encoder_only=True).to('cuda')
        checkpoint = torch.load(ddpm_ckpt)
        self.diffmodel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.diffmodel.eval()

        self.transform = lambda x : (x * 2) - 1

   
    def forward(self, lowres_input): # remember to use lowres input of size 128x128
        params = self.params
        bs = lowres_input.shape[0]
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']

        x = lowres_input
        # print('lowres', x.shape)

        timestep = torch.full((1,), 0, device=x.device, dtype=torch.long)
        # print('t', timestep.shape)

        x = self.diffmodel(self.transform(x), timestep)
        # print('unet', x.shape)

        x = self.splat_addn_layer(x)
        ###
        # splat_features = x
        # # print(f'splat features: {splat_features.shape}')

        # # # print(x.shape)                                              # [4, 3, 256, 256]
        # # for layer in self.splat_features:
        # #     x = layer(x)
        # # splat_features = x
        # # # print(f'splat features: {splat_features.shape}')            # [4, 64, 16, 16]
        
        x2 = lowres_input
        for layer in self.splat_features:
            x2 = layer(x2)

        x = self.splat_cct_layer(torch.cat([x, x2], dim=1))
        splat_features = x

        ###
        
        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        # print(f'global features: {x.shape}')                        # [4, 1024]

        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x
        # print(f'global features fc: {global_features.shape}')       # [4, 64]

        x = splat_features
        for layer in self.local_features:
            x = layer(x)        
        local_features = x
        # print(f'local features: {local_features.shape}')            # [4, 64, 16, 16]


        fusion_grid = local_features
        fusion_global = global_features.view(bs,8*cm*lb,1,1)
        # print(fusion_grid.shape, fusion_global.shape)               # [4, 64, 16, 16], [4, 64, 1, 1]
        fusion = self.relu( fusion_grid + fusion_global )       

        # print(fusion.shape)                                         # [4, 64, 16, 16]

        z = self.depth_param_predictor(fusion)
        z = z.reshape(z.shape[0], -1)
        z = self.depth_param_predictor_fc(z)

        x = self.conv_out(fusion)
        # print(x.shape)                                              # [4, 96, 16, 16]
        s = x.shape
        y = torch.stack(torch.split(x, self.nin*self.nout, 1),2)
        # y = torch.stack(torch.split(y, self.nin, 1),3)
        # print('y:', y.shape)                                        # [4, 12, 8, 16, 16]
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        # print(x.shape)
        return y, z



class HDRPointwiseNN_depth(nn.Module):

    def __init__(self, params,ckpt):
        super(HDRPointwiseNN_depth, self).__init__()
        self.coeffs = Coeffs(params=params,ddpm_ckpt=ckpt)
        self. guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        # self.bsa = bsa.BilateralSliceApply()

        # self.a = nn.Parameter(torch.tensor(1.0))
        # self.b = nn.Parameter(torch.tensor(1.0))
        # self.c = nn.Parameter(torch.tensor(1.0))
        # self.d = nn.Parameter(torch.tensor(1.0))


    def forward(self, lowres, fullres):
        coeffs, z_params = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        # out = bsa.bsa(coeffs,guide,fullres)

        return out, z_params

    def forward_dual(self, lowres, fullres, gt):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, gt)
        # out = bsa.bsa(coeffs,guide,fullres)
        return out


#########################################################################################################
