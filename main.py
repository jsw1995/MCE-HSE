## 最终版本的会议论文版本

import argparse
import math
import os
# import cv2 as cv
import cv2 as cv
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import skimage
from matplotlib import pyplot as plt
import hashlib
import time
import pywt
from utils import encryption, dencryption
from Analysis_function import mssim, psnr


if __name__ == '__main__':

    # im = cv.imread('squat/lena_gray_512.tif', 0)
    # cover = cv.imread('squat/mandril_gray.tif', 0)

    im = cv.imread('date/peppers_gray.tif', 0)
    cover = cv.imread('date/woman.tif', 0)

    [m, n]=cover.shape
    sec_key = [0.55, 0.55, 21, 21]
    T = 20
    d = -3

    cip, maxx3, minx3, TT, dynamic_key = encryption(im,cover,sec_key,T,d)
    rim = dencryption(cip,cover, maxx3, minx3, TT, dynamic_key, im.shape)

    mssim_cip = mssim(cip,cover,[m-8,n-8])
    psnr_cip = psnr(cip,cover)
    print(mssim_cip,psnr_cip)

    mssim_im = mssim(im,rim,[m-8,n-8])
    psnr_im = psnr(im,rim)
    print(mssim_im,psnr_im)

    cv.imshow('im', im)
    cv.imshow('cip', np.uint8(cip))
    cv.imshow('rim', np.uint8(rim))
    plt.show()
    cv.waitKey(0)

