import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset , ConcatDataset , DataLoader
from load_raw import preprocess_depthmap, load_depthmap
from torchvision.transforms import functional as F
import numpy as np
from load_raw import estimate_far, load_image
import argparse

def load_image(image_path, max_side = 1024):
    image_file = Image.open(image_path)
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.float64(image_file) / 255.0

class ImageSet(Dataset):
    def __init__(self,data_dir ,params=None, suffix='', aug=False):
        self.suffix = suffix
        self.aug = aug
        self.video_dir = f"{data_dir}/train"
        self.gt_dir =f"{data_dir}/gt"

        self.in_files = os.listdir(f"{self.video_dir}")

        ls = params['net_input_size']
        fs = params['net_output_size']
        self.ls, self.fs = ls, fs
        self.low = transforms.Compose([
            transforms.Resize((ls,ls), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.correction = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0),
        ])
        self.out = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.full = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.in_files)-1

    def __getitem__(self, idx):
        fname = self.in_files[idx]
        imagein = Image.open(f"{self.video_dir}/{fname}").convert('RGB')
        gt = Image.open(f"{self.gt_dir}/{fname}").convert('RGB')

        if self.aug:
            imagein = self.correction(imagein)
            next_imagein = self.correction(next_imagein)
        imagein_low = self.low(imagein)
        imagein_full = self.full(imagein)
        target = self.full(gt)
        return imagein_low, imagein_full,target
