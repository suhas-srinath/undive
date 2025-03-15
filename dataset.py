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

class VideoSet(Dataset):
    def __init__(self,video_dir,gt_dir ,forward_flow_dir ,backward_flow_dir,params=None, suffix='', aug=False):
        self.suffix = suffix
        self.aug = aug
        self.video_dir = video_dir
        self.forward_flow_dir = forward_flow_dir
        self.backward_flow_dir = backward_flow_dir
        self.gt_dir =gt_dir

        self.in_files = sorted(os.listdir(f"{self.forward_flow_dir}/high_flow"))

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
        index = fname[:4]
        next_index = str(int(fname[:4])+1).zfill(4)
        imagein = Image.open(f"{self.video_dir}/{index}.png").convert('RGB')
        next_imagein = Image.open(f"{self.video_dir}/{next_index}.png").convert('RGB')
        gt = Image.open(f"{self.gt_dir}/{fname}").convert('RGB')

        if self.aug:
            imagein = self.correction(imagein)
            next_imagein = self.correction(next_imagein)
        imagein_low = self.low(imagein)
        imagein_full = self.full(imagein)
        next_imagein_low = self.low(next_imagein)
        next_imagein_full = self.full(next_imagein)
        target = self.full(gt)

        forward_flow_low = np.load(f"{self.forward_flow_dir}/low_flow/{index}.npy").astype(np.float32).transpose([2,0,1])
        forward_flow_full = np.load(f"{self.forward_flow_dir}/high_flow/{index}.npy").astype(np.float32).transpose([2,0,1])

        backward_flow_low = np.load(f"{self.backward_flow_dir}/low_flow/{next_index}.npy").astype(np.float32).transpose([2,0,1])
        backward_flow_full = np.load(f"{self.backward_flow_dir}/high_flow/{next_index}.npy").astype(np.float32).transpose([2,0,1])

        return imagein_low, imagein_full,next_imagein_low, next_imagein_full, forward_flow_low, forward_flow_full, backward_flow_low, backward_flow_full ,target


class trainDataset(Dataset):
    def __init__(self,data_dir ,params=None, suffix='', aug=False):

        def dataset(data_dir,params,suffix,aug):
            datasets = []
            for video in os.listdir(f"{data_dir}/frames"):
                for gt in os.listdir(f"{data_dir}/gt"):
                    if gt in video:
                        gt_video = gt
                        break

                video_dir = f"{data_dir}/bsr_images/{video}"
                gt_dir = f"{data_dir}/gt/{gt_video}"
                forward_flow_dir = f"{data_dir}/forward_flow/{video}"
                backward_flow_dir = f"{data_dir}/backward_flow/{video}"
                datasets.append(VideoSet(
                    video_dir,
                    gt_dir,
                    forward_flow_dir,
                    backward_flow_dir,
                    params ,
                    suffix,
                    aug
                ))
            return datasets

        self.datasets = dataset(data_dir,params,suffix,aug)
        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length

