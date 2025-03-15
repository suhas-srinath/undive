import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet_v2 import FastFlowNet
from flow_vis import flow_to_color
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str, required=True, help='Path for Input Images')
parser.add_argument('--hflow-path' ,type = str , required=True ,help = "path for high flow")
parser.add_argument('--lflow-path' ,type = str , required=True ,help = "path for low flow")
args = parser.parse_args()
img_path = args.image_path
high_path = args.hflow_path
low_path = args.lflow_path

from tqdm import tqdm
import os

div_flow = 20.0
div_size = 64

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('/media/suhas/Data/uwe/llve/FastFlowNet/checkpoints/fastflownet_ft_mix.pth'))


img_files = sorted(os.listdir(img_path))


for img_index in tqdm(img_files[:-1:2]):
    img_index = img_files.index(img_index)
    img1_path = os.path.join(img_path, img_files[img_index+1])
    img2_path = os.path.join(img_path, img_files[img_index])

    # Orig Resolution

    img1 = torch.from_numpy(cv2.resize(cv2.imread(img1_path), (512,512))).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.from_numpy(cv2.resize(cv2.imread(img2_path), (512,512))).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    output = model(input_t).data

    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    flow = flow[0].cpu().permute(1, 2, 0).numpy()

    np.save(high_path+'/{}.npy'.format(img_files[img_index].split('.')[0]), flow)

    # Low Res

    img1 = torch.from_numpy(cv2.resize(cv2.imread(img1_path), (256,256))).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.from_numpy(cv2.resize(cv2.imread(img2_path), (256,256))).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    output = model(input_t).data

    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    flow = flow[0].cpu().permute(1, 2, 0).numpy()

    np.save(low_path+'/{}.npy'.format(img_files[img_index].split('.')[0]), flow)
