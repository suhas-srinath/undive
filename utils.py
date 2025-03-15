import numpy as np
import cv2
import os
import glob

import torch, torchvision

def resize(img, size=512, strict=False):
    short = min(img.shape[:2])
    scale = size/short
    if not strict:
        img = cv2.resize(img, (round(
            img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    return img

def crop(img, size=512):
    try:
        y, x = random.randint(
            0, img.shape[0]-size), random.randint(0, img.shape[1]-size)
    except Exception as e:
        y, x = 0, 0
    return img[y:y+size, x:x+size, :]


def load_image(filename, size=None, use_crop=False):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = resize(img, size=size)
    if use_crop:
        img = crop(img, size)
    return img

def get_latest_ckpt(path):
    try:
        list_of_files = glob.glob(os.path.join(path,'*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None

def save_params(state, params):
    state['model_params'] = params
    return state

def load_params(state):
    params = state['model_params']
    del state['model_params']
    return state, params

def tv_loss(input, output, weight):
    I = torchvision.transforms.functional.rgb_to_grayscale(input)
    L = torch.log(I + 0.0001)

    L = L.permute(0, 2, 3, 1)
    output = output.permute(0, 2, 3, 1)

    dx = L[:, :-1, :-1, :] - L[:, :-1, 1:, :]
    dy = L[:, :-1, :-1, :] - L[:, 1:, :-1, :]

    alpha = torch.tensor(1.2)
    lamda = torch.tensor(1.5)
    dx = torch.div(lamda, torch.pow(torch.abs(dx),alpha) + torch.tensor(0.0001))
    dy = torch.div(lamda, torch.pow(torch.abs(dy),alpha) + torch.tensor(0.0001))

    shape = output.size()

    x_loss = dx *((output[:, :-1, :-1, :] - output[:, :-1, 1:, :])**2)
    y_loss = dy *((output[:, :-1, :-1, :] - output[:, 1:, :-1, :])**2)

    tvloss = torch.mean(x_loss + y_loss)/2.0

    return tvloss * weight

def cos_loss(t, out):
    epsilon=1e-7
    return torch.mean(torch.acos(torch.clamp(torch.sum(torch.mul(torch.nn.functional.normalize(t, p=2, dim=1), torch.nn.functional.normalize(out, p=2, dim=1)), dim=1), -1.0 + epsilon, 1.0 - epsilon)))

def angle(v1, v2):
    def unit(v):
      return v/torch.linalg.norm(v)
    v1_u = unit(v1)#/255.0)
    v2_u = unit(v2)#/255.0)
    return torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))
