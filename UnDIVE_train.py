import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from warp import WarpingLayerBWFlow

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import cv2
import random

import skimage.exposure
from torchvision import transforms
import glob
import pyiqa

from utils import load_image, resize, tv_loss, cos_loss
from dataset import trainDataset
from model_depth import HDRPointwiseNN_depth, L2LOSS

start = time.time()

def test(ckpt, params = {}, epoch={}):
    print("##################################TESTING#####################################")
    model_test = HDRPointwiseNN_depth(params=params,ckpt=params['ddpm_ckpt'])
    model_test.load_state_dict(torch.load(ckpt))
    device = torch.device("cuda")

    model_test.eval()
    model_test.to(device)

    tensor = transforms.Compose([transforms.ToTensor(),])
    
    for video in os.listdir(args.test_video):
        test_files = glob.glob(f'{args.test_video}/{video}/*')
        os.makedirs(f"{args.test_out}/{video}" , exist_ok = True)
        for img_path in test_files:
            img_name = img_path.split('/')[-1]
            low = tensor(resize(load_image(img_path),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
            full = tensor(load_image(img_path).astype(np.float32)).repeat(1,1,1,1)/255

            low = low.to(device)
            full = full.to(device)
            with torch.no_grad():
                res, _ = model_test(low, full)
                res = torch.clip(res, min=full, max=torch.ones(full.shape).to(device))
                img =  torch.div(full, torch.add(res, 0.001))
                res = (res.cpu().detach().numpy()).transpose(0,2,3,1)[0]
                res = skimage.exposure.rescale_intensity(res, out_range=(0.0,255.0)).astype(np.uint8)

                img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
                img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
                os.makedirs(f"{params['test_out']}/{video}/{epoch.zfill(3)}" , exist_ok = True)
                cv2.imwrite(f"{params['test_out']}/{video}/{epoch.zfill(3)}/{img_name}", img[...,::-1])

def train(video_train_loader , optimizer , model , l1 ,ssim_loss , count):
    for i, (low, full,next_low , next_full ,forward_flow_low, forward_flow_full ,backward_flow_low,backward_flow_full, target) in enumerate(video_train_loader):

        low, full,next_low , next_full, forward_flow_low, forward_flow_full,backward_flow_low, backward_flow_full, target = low.to(device), full.to(device),next_low.to(device), next_full.to(device), forward_flow_low.to(device), forward_flow_full.to(device) , backward_flow_low.to(device), backward_flow_full.to(device),target.to(device)
        optimizer.zero_grad()


        # task loss
        illum, z_params = model(low, full)
        res = torch.clip(illum, min=full, max=torch.ones(full.shape).to(device))
        pred =  torch.div(full, torch.add(res, 0.001))

        color = args.color_wt * cos_loss(pred, target)
        smooth = args.smooth_wt * tv_loss(full, res, 1)
        recon = args.recon_wt * (l1(pred, target) * 0.15 + (1 - ssim_loss(pred, target)) * 0.85)
        loss =  recon + color + smooth

        loss_t = 0

        # temporal loss forward
        if args.temp != 0:
            low_t = warp(low, forward_flow_low)
            full_t = warp(full, forward_flow_full)

            illum_t, z_params_t = model(low_t, full_t)
            res_t = torch.clip(illum_t, min=full_t, max=torch.ones(full_t.shape).to(device))
            input_t_pred =  torch.div(full_t, torch.add(res_t, 0.001))

            pred_t = warp(pred, forward_flow_full)
            
            loss_t_f = l1(input_t_pred, pred_t)

        # temporal loss backward
        if args.temp == 2:

            illum, z_params = model(next_low, next_full)
            next_res = torch.clip(illum, min=next_full, max=torch.ones(next_full.shape).to(device))
            next_pred =  torch.div(next_full, torch.add(next_res, 0.001))


            low_t = warp(next_low, backward_flow_low)
            full_t = warp(next_full, backward_flow_full)

            illum_t, z_params_t = model(low_t, full_t)
            res_t = torch.clip(illum_t, min=full_t, max=torch.ones(full_t.shape).to(device))
            input_t_pred =  torch.div(full_t, torch.add(res_t, 0.001))

            pred_t = warp(next_pred, backward_flow_full)
            
            loss_t_b = l1(input_t_pred, pred_t)

            loss_t = 0.5 * loss_t_f + 0.5 * loss_t_b

        loss_t = loss_t * args.temp_wt * bool(args.temp)

        total_loss = 0.5 * loss + loss_t

        total_loss.backward()
	     		
        if (count+1) % params['log_interval'] == 0:
            tloss = total_loss.item()
            Loss1 = loss.item()
            Loss2 = 0 if args.temp == 0 else loss_t.item()
            t_f = 0  if params['temp'] ==0 else loss_t_f.item()
            t_b = loss_t_b.item() if params['temp']==2 else 0
            lc = color.item()
            lr = recon.item()
            ls = smooth.item()
            batch = (count + 1)%len(video_train_loader)
            print(f'Epoch: {e} [{batch}/{len(video_train_loader)}]\t loss={tloss:.5f} l1={Loss1:.5f} l2={Loss2:.5f}  \tTemporal Loss : forward = {t_f:.5f} backward = {t_b:.5f}  \tTask Loss : color = {lc:.5f} recon = {lr:.5f} smooth = {ls:.5f}' , flush = True) 
            print(f'Epoch: {e} [{batch}/{len(video_train_loader)}]\t loss={tloss:.5f} l1={Loss1:.5f} l2={Loss2:.5f}  \tTemporal Loss : forward = {t_f:.5f} backward = {t_b:.5f}  \tTask Loss : color = {lc:.5f} recon = {lr:.5f} smooth = {ls:.5f}' , flush = True , file = log_file)
        
        optimizer.step()
    
        count+=1

# Parse arguments
parser = argparse.ArgumentParser(description='TC-HDRNet')
parser.add_argument('--video-data-path', type=str, help='path to the video dataset') #Required
parser.add_argument('--epochs', default=100, type=int, help='number of epochs (default: 100)')
parser.add_argument('--bs', default=1, type=int, help='batch size(default: 1)')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate (default: 1e-4)')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (default: 0)')
parser.add_argument('--log', type=str,default = "Log", help='Path to Log.') #Required 

parser.add_argument('--color-wt', default=0.1, type=float, help='Weight of Color Loss(default: 0.1)')
parser.add_argument('--smooth-wt', default=0.2, type=float, help='Weight of Smoothness Loss(default: 0.2)')
parser.add_argument('--recon-wt', default=1, type=float, help='Weight of Reconstruction Loss(default: 1)')
parser.add_argument('--temp-wt' , default = 20 , type = int , help="Weight for Temporal Consistancy Loss(default: 20)")
parser.add_argument('--temp', default=2, type=int, help='Temporal Consistancy Type\n0: none\n1:unidirectional\n2:bidirectional')

parser.add_argument('--ddpm-ckpt' , type = str, default = "PretrainedModels/DDPM_100.pth", help='Path to ddpm ckpt(default : Pretrained_Models/DDPM_100.pth)')
parser.add_argument('--ckpt-path' , type = str, default = "PretrainedModels/UIEB_pretrain_150.pth", help='Path to prev ckpt(default : Pretrained_Models/UIEB_pretrain_150.pth)')
parser.add_argument('--exp' , type = str , default = "UnDIVE_Train" , help='Experiment Name(default : UnDIVE_Train)')
parser.add_argument('--test-video', type=str, help='TestVideo directory Path') #Required
parser.add_argument('--test-out', type=str,default = "Output", help='Output path') #Required

parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--ckpt-interval', type=int, default=10)

parser.add_argument('--luma-bins', type=int, default=8)
parser.add_argument('--channel-multiplier', default=1, type=int)
parser.add_argument('--spatial-bin', type=int, default=16)
parser.add_argument('--guide-complexity', type=int, default=16)
parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

args = parser.parse_args()
params = vars(parser.parse_args())
os.makedirs(args.log , exist_ok = True)
os.makedirs(args.test_out , exist_ok = True)
log_file = open(f"{args.log}/{args.exp}.txt" , 'w')
log_dir = args.log
print(f"color : {args.color_wt}",flush = True)
print(f"smooth : {args.smooth_wt}",flush = True)
print(f"recon : {args.recon_wt}",flush = True)
print(f"temp : {args.temp}",flush = True)


print(f"color : {args.color_wt}",file = log_file , flush = True)
print(f"smooth : {args.smooth_wt}",file = log_file ,flush = True)
print(f"recon : {args.recon_wt}",file = log_file ,flush = True)
print(f"temp : {args.temp}",file = log_file ,flush = True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda")

torch.manual_seed(ord('c')+137)
random.seed(ord('c')+137)
np.random.seed(ord('c')+137)


datasets = trainDataset(data_dir = args.video_data_path, params=params)
video_train_loader = DataLoader(datasets, batch_size=args.bs, shuffle=True ,num_workers = 4 , pin_memory = True)

model = HDRPointwiseNN_depth(params=params,ckpt=params['ddpm_ckpt'])

if params['ckpt_path']:
    print('Loading previous state:', params['ckpt_path'])
    model.load_state_dict(torch.load(params['ckpt_path']))

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

l1 = torch.nn.L1Loss()
ssim_loss = pyiqa.create_metric('ssimc', device=device, as_loss=True)
warp = WarpingLayerBWFlow().cuda()

model.train()
count =0
os.makedirs(params['test_out'] , exist_ok = True)
for e in range(0 , params['epochs']+1):
    train(video_train_loader , optimizer , model , l1 ,ssim_loss, count)
    if e % args.ckpt_interval == 0 or e==10:
        os.makedirs(f"{log_dir}/ckpt" , exist_ok = True)
        torch.save(model.state_dict() , f"{log_dir}/ckpt/undive_ckpt_{e}.pth")
        model.eval()
        with torch.no_grad():
            test(os.path.join(f"{log_dir}/ckpt/undive_ckpt_{e}.pth"), params=params, epoch=str(e))


torch.save(model.state_dict() , f"{log_dir}/final.pth")

min , sec = divmod(time.time() - start , 60)
hour , min = divmod(min , 60)

print(f"total time : {hour} hours {min} mins and {sec} secs" , flush = True)
print(f"total time : {hour} hours {min} mins and {sec} secs" , flush = True ,file = log_file)


log_file.close()
'''
python UnDIVE_train.py --video-data-path "" --test-video ""
'''
