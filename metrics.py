import torch
import numpy as np

import os
import cv2
import sys
import math
from skimage import io, color, filters
from skimage.metrics import structural_similarity
from PIL import Image
from scipy import ndimage

def psnr(target, prediction):
    x = (target-prediction)**2
    x = x.view(x.shape[0], -1)
    p = torch.mean((-10/np.log(10))*torch.log(torch.mean(x, 1)))
    return p

def ref_based(out, ref):
    """
    Returns the PSNR, PSNR on L Channel (prescribed by FuNiE-Gan) and SSIM.
    PCQI Implementation on MATLAB.
    """

    mse = np.mean((out-ref)**2)
    psnr_metric = 10*math.log10(1/mse)
    ssim_metric = structural_similarity(out,ref,channel_axis=-1,data_range=1.0)

    out = np.array(Image.fromarray(np.uint8(255 * out)).convert("L"), dtype=np.float64)
    ref = np.array(Image.fromarray(np.uint8(255 * ref)).convert("L"), dtype=np.float64)

    rmse = math.sqrt(np.mean((out-ref).flatten('C') ** 2.))
    psnr_L = 20*math.log10(255.0/rmse)

    return psnr_metric, psnr_L, ssim_metric

def non_ref_based(im):
    """
    Returns the UIQM and UCIQE.
    CCF Implementation on MATLAB.
    """

    uiqm = getUIQM(np.array(Image.fromarray(np.uint8(255*im))))
    # uiqm = getUIQM(np.array(Image.fromarray(np.uint8(255*im)).resize((256, 256)))) #FuNie-Gan does this
    uciqe = getUCIQE(np.array(Image.fromarray(np.uint8(255*im))))
    return uiqm, uciqe

# UIQM Utility

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    x = sorted(x)     # sort pixels by intensity - for clipping
    K = len(x)    # get number of pixels
    T_a_L = math.ceil(alpha_L*K)    # calculate T alpha L and T alpha R
    T_a_R = math.floor(alpha_R*K)
    weight = (1/(K-T_a_L-T_a_R))    # calculate mu_alpha weight
    s   = int(T_a_L+1)    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    k1 = x.shape[1]/window_size    # if 4 blocks, then 2x2...etc.
    k2 = x.shape[0]/window_size
    w = 2./(k1*k2)    # weight
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0: val += 0            # bound checks, can't do log(0)
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    Rs = sobel(R) # first apply Sobel edge detector to each RGB components
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)    # multiply the edges detected for each channel by the channel itself
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 10)    # get eme for each channel
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    lambda_r = 0.299    # coefficients
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    k1 = x.shape[1]/window_size    # if 4 blocks, then 2x2...etc.
    k2 = x.shape[0]/window_size
    w = -1./(k1*k2)    # weight
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
    return w*val

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
      https://ieeexplore.ieee.org/abstract/document/7305804
    """
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

# UCIQE Utility

def getUCIQE(image):
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.

    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val