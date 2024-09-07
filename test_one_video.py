import argparse
import os
from tqdm import tqdm
import time
import numpy as np
import torch
import cv2
import random

import skimage.exposure
from torchvision import transforms
import glob

from utils import load_image, resize
from model_depthv5 import HDRPointwiseNN_depthv2


def test(ckpt, params = {}):
    model2 = HDRPointwiseNN_depthv2(params=params,ckpt=params['ddpm_ckpt'])
    model2.load_state_dict(torch.load(ckpt),strict=False)
    device = torch.device("cuda")

    model2.eval()
    model2.to(device)
    tensor = transforms.Compose([
        transforms.ToTensor(),])

    if os.path.isdir(params['test_video']):
        test_files = glob.glob(params['test_video']+'/*')
    else:
        test_files = [params['test_video']]

    for img_path in tqdm(test_files):
        start = time.time()
        img_name = img_path.split('/')[-1]
        low = tensor(resize(load_image(img_path),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
        full = tensor(load_image(img_path).astype(np.float32)).repeat(1,1,1,1)/255                           
        low = low.to(device)
        full = full.to(device)
        with torch.no_grad():
            res, _ = model2(low, full)
            res = torch.clip(res, min=full, max=torch.ones(full.shape).to(device))
            img =  torch.div(full, torch.add(res, 0.001))
            res = (res.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            res = skimage.exposure.rescale_intensity(res, out_range=(0.0,255.0)).astype(np.uint8)
            img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['test_out'],img_name), img[...,::-1])
       
if __name__ == "__main__":
# Parse arguments

    # Inference Params
    parser = argparse.ArgumentParser(description='TC-HDRNet')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (default: 0)')
    parser.add_argument('--checkpoint', default='Checkpoints/Un-DIVE_checkpoint.pth' , type=str, help='Path to Un-DIVE checkpoint')
    parser.add_argument('--ddpm-ckpt',default="Checkpoints/ddpm_checkpoint_100.pth",type=str, help="Path toDDPM Checkpoint")
    parser.add_argument('--test-out', type=str, default='Outputs/Frames', dest="test_out", help='Path to Output')
    parser.add_argument('--test-video', type=str,default='Data/BackscatterdRemoved_Images', dest="test_video", help='Path to BackScattered Video Frames')

    # Model Params
    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--guide-complexity', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

    args = parser.parse_args()
    params = vars(parser.parse_args())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")

    torch.manual_seed(ord('c')+137)
    random.seed(ord('c')+137)
    np.random.seed(ord('c')+137)
    os.makedirs(params["test_out"],exist_ok=True)
    test(params["checkpoint"] , params=params)
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{params['test_out']}/*.png' -c:v libx264 -pix_fmt yuv420p video.mp4")
