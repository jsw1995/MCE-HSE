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
import scipy



def ICM(initial, parameters, N):  # 混沌
    x = np.zeros([N + 1001, 1])
    y = np.zeros([N + 1001, 1])
    x[0] = initial[0]
    y[0] = initial[1]
    a = parameters[0]
    b = parameters[1]
    for i in range(N + 1000):
        x[i + 1] = math.sin(a / y[i]) * math.sin(b / x[i])
        y[i + 1] = math.sin(a / x[i]) * math.sin(b / y[i])

    return x[1001:N + 1001], y[1001:N + 1001]

def hex_dec(str2):  # 十六转十
    b = eval(str2)
    return b

def dec_hex(str1):  # 十转十六 （输入为str类型）
    a = str(hex(eval(str1)))
    return a

def dec2bin(a):
    if a.ndim == 3:
        [pages, rows, columns] = a.shape
    else:
        [rows, columns] = a.shape
        pages = 1

    a = a.reshape(pages * rows * columns, 1)

    b = a // 128
    a = a - b * 128
    b1 = a // 64
    a = a - b1 * 64
    b2 = a // 32
    a = a - b2 * 32
    b3 = a // 16
    a = a - b3 * 16
    b4 = a // 8
    a = a - b4 * 8
    b5 = a // 4
    a = a - b5 * 4
    b6 = a // 2
    a = a - b6 * 2
    c = np.concatenate((b, b1, b2, b3, b4, b5, b6, a), axis=1)
    return c

def bin2dec(tem13):
    tem11 = np.array([[128], [64], [32], [16], [8], [4], [2], [1]])
    tem12 = np.dot(tem13, tem11)
    return tem12

def im_sha256(im, sec_key):
    # 当前时间与输入图像的哈希值
    # 动态密钥产生
    if len(im.shape) == 3:
        H1 = hashlib.sha256(im).hexdigest()
    else:
        H1 = hashlib.sha256(im).hexdigest()

    H = np.zeros([32, ])
    for i in range(32):
        H[i] = eval('0x' + H1[2 * i:2 * (i + 1)])

    H = np.array(H, np.uint8)
    h = np.sum(H) / 8192

    h21 = 0
    h22 = 0
    h23 = 0
    h24 = 0
    for i in range(8):
        h21 = h21 ^ H[i]
        h22 = h22 ^ H[i+8]
        h23 = h23 ^ H[i + 16]
        h24 = h24 ^ H[i + 24]


    x0 = (h21/127.5+sec_key[0])*h/2
    y0 = (h22/127.5+sec_key[1])*h/2
    a = h23/25.5 * h / 2 + sec_key[2]
    b = h24/25.5 * h / 2 + sec_key[3]

    return [x0, y0, a, b]

def DWT(N):
    a = pywt.Wavelet('sym8')
    h = np.expand_dims(np.array(a.dec_lo),axis=0)
    g = np.expand_dims(np.array(a.dec_hi),axis=0)
    L = h.shape[1]
    rank_max = int(math.log2(N))
    rank_min = int(math.log2(L))+1
    ww = np.eye(N)

    for jj in range(rank_min,rank_max+1):

        nn = 2 ** jj
        p1_0 = np.concatenate((h,np.zeros((1, nn-L))),axis=1)
        p2_0 = np.concatenate((g,np.zeros((1, nn-L))),axis=1)

        p1 = np.zeros((int(nn / 2), p1_0.shape[1]))
        p2 = np.zeros((int(nn / 2), p1_0.shape[1]))

        for ii in range(0,int(nn/2)):
            p1[ii,:] = np.roll(p1_0.T,2*(ii-1),axis=0).T
            p2[ii,:] = np.roll(p2_0.T,2*(ii-1),axis=0).T

        w1 = np.concatenate((p1,p2),axis=0)
        mm = 2 ** rank_max - len(w1)
        w11 = np.concatenate((w1,np.zeros((len(w1),mm))),axis=1)
        w12 = np.concatenate((np.zeros((mm,len(w1))),np.eye(mm)),axis=1)
        w = np.concatenate((w11,w12),axis=0)
        ww = ww.dot(w)

    return ww

def nsl0(y,A):
    sigma_min = 0.1   #0.1
    sigma_decrease_factor = 0.98  #0.95
    ksai = 0.1   #0.1

    A_pinv = np.linalg.pinv(A)
    s = A_pinv.dot(y)

    sigma = 4 * np.max(np.abs(s))
    r = 0
    r0 = y - A.dot(s)

    while (sigma>sigma_min):

        if np.sum((r-r0)**2) < ksai:
            d = -(s * sigma*sigma) / (s*s + sigma*sigma)
            s = s + d
            s = s - np.dot(A_pinv, (np.dot(A, s) - y))
            r0 = y - A.dot(s)
        sigma = sigma*sigma_decrease_factor
    return s

def scramble(im,r):
    """
    :param im: 输入图像
    :param r: 随机序列大小与图像大小一样
    :return: 置乱后的图像cip
    """
    m,n = im.shape
    T1 = np.array(r[0:m*n].ravel().argsort(), int)  # 序列排序 排序后返回相应的索引
    tem01 = im.reshape(1, m * n)  # 将height×width的二维矩阵转换为1×height*width形式的矩阵
    cip = np.zeros([1, m * n])
    for i in range(m * n):
        cip[0, i] = tem01[0, T1[i]]

    cip = cip.reshape(m, n)
    return cip

def den_scramble(cip,r):
    """
    :param cip: 置乱图像
    :param r: 随机序列
    :return: 解置乱图像rim
    """
    m, n = cip.shape
    T1 = np.array(r[0:m*n].ravel().argsort(), int)  # 序列排序 排序后返回相应的索引
    tem01 = cip.reshape(1, m * n)
    rim = np.zeros([1, m * n])
    for i in range(m * n):
        rim[0, T1[i]] = tem01[0, i]

    rim = rim.reshape(m, n)
    return rim

def compression(image,r1,r2,T):
    if len(image.shape) == 3:
        [rows1, columns1, hight] = image.shape
        ww = DWT(rows1)
        X1r = ww.dot(image[:, :, 0]).dot(ww.T)
        X1g = ww.dot(image[:, :, 1]).dot(ww.T)
        X1b = ww.dot(image[:, :, 2]).dot(ww.T)
        X1 = np.concatenate((X1r, X1g, X1b), axis=0)
        [rows, columns] = X1.shape
    else:
        [rows, columns] = image.shape
        ww = DWT(rows)
        X1 = ww.dot(image).dot(ww.T)

    XXX = sorted(np.abs(X1).ravel())
    T = XXX[int(rows * columns * (1 - 0.05))]  # 文章中RS算出来为0.05.直接用
    print(T)

    X1[np.abs(X1) < T] = 0
    X2 = scramble(X1,r1)
    R = np.reshape(r2[0: int(rows * 0.25 * rows)], [int(rows * 0.25), rows])
    R[R <= 0] = -1
    R[R > 0] = 1
    X3 = R.dot(X2)
    max_x3, min_x3 = np.max(X3), np.min(X3)
    X3 = np.round((X3 - min_x3) / (max_x3 - min_x3) * 255)

    return X3, max_x3, min_x3

def refactor(image,r1,r2,max_x3, min_x3,plaintext_shape):

    if len(plaintext_shape) == 3:
        rows = 3 * plaintext_shape[0]
        columns = plaintext_shape[1]
    else:
        rows = plaintext_shape[0]
        columns = plaintext_shape[1]

    X3 = image / 255 * (max_x3 - min_x3) + min_x3
    R = np.reshape(r2[0: int(rows * 0.25 * rows)], [int(rows * 0.25), rows])
    R[R <= 0] = -1
    R[R > 0] = 1
    rec = nsl0(X3, R)  ##
    X4 = rec.reshape(rows, columns)
    X5 = den_scramble(X4, r1)
    # 小波逆变换
    if len(plaintext_shape) == 3:
        [rows1, columns1, hight1] = plaintext_shape
        ww = DWT(rows1)
        rimr = ww.T.dot(X5[0:rows1, :]).dot(ww)
        rimg = ww.T.dot(X5[rows1:2 * rows1, :]).dot(ww)
        rimb = ww.T.dot(X5[2 * rows1:3 * rows1, :]).dot(ww)
        rim = np.concatenate((np.expand_dims(rimr, axis=2), np.expand_dims(rimg, axis=2), np.expand_dims(rimb, axis=2)),
                             axis=2)
    else:
        ww = DWT(rows)
        rim = ww.T.dot(X5).dot(ww)

    return np.clip(rim, 0, 255)

def embed2(image,cover, d):
    """
    嵌入操作
    :param image: 二维
    :param cover: 可三维可二维
    :return: 与cover相近的密文
    """
    aa = np.zeros((255,))
    for x_value in range(255):
        aa[x_value] = float(image[image == x_value].shape[0])

    TT = aa.argmax() + d
    X33 = np.copy(image)
    X33[image < TT] = X33[image < TT] - TT + 256
    X33[image >= TT] = X33[image >= TT] - TT

    [rows, columns] = image.shape
    bit_cover = dec2bin(cover)
    bit_x3 = np.reshape(dec2bin(X33),[int(rows*columns*4), 2])
    cip_bit = np.copy(bit_cover)
    cip_bit[0:int(rows*columns*4), :] = np.concatenate((cip_bit[0:int(rows*columns*4), 0:6], bit_x3.astype(int) ^ bit_cover[0:int(rows*columns*4), 6:8].astype(int)), axis=1)

    if len(cover.shape) == 3:
        [cover_k,cover_m,cover_n] = cover.shape
        cip = np.reshape(bin2dec(cip_bit), [cover_k,cover_m,cover_n])
    else:
        [cover_m, cover_n] = cover.shape
        cip = np.reshape(bin2dec(cip_bit), [cover_m, cover_n])

    cip1 = np.copy(cip).astype(int)
    cip1[cip - cover == 3] = cip[cip - cover == 3] - 4
    cip1[cip - cover == -3] = cip[cip - cover == -3] + 4
    cip1[cip1 > 255] = cip[cip1 > 255]
    cip1[cip1 < 0] = cip[cip1 < 0]

    return cip1, TT

def extract2(cip,cover,image_shape, T):
    """
    :param image_shape: 类噪声密文的尺寸
    """
    [rows, columns] = image_shape
    bit_cip = dec2bin(cip)
    bit_cover = dec2bin(cover)
    tcip_bit = bit_cip[0:int(rows*columns*4), 6:8].astype(int) ^ bit_cover[0:int(rows*columns*4), 6:8].astype(int)
    tcip = bin2dec(np.reshape(tcip_bit, [int(rows*columns), 8]))
    tcip = np.reshape(tcip,[rows, columns])

    X33 = np.copy(tcip)
    X33[tcip < 256 - T] = X33[tcip < 256 - T] + T
    X33[tcip >= 256 - T] = X33[tcip >= 256 - T] + T - 256

    return X33

def encryption(im, cover, sec_key,T,d):
    [m,n]=im.shape
    dynamic_key = im_sha256(im, sec_key)
    x, y = ICM(initial=dynamic_key[0:2], parameters=dynamic_key[2:4], N=m*n)
    X3, maxx3, minx3 = compression(im, x, y, T)
    cip, TT = embed2(X3, cover, d)

    return cip, maxx3, minx3, TT, dynamic_key

def dencryption(cip, cover, maxx3, minx3, TT, dynamic_key, im_shape):
    [m,n]=im_shape
    x, y = ICM(initial=dynamic_key[0:2], parameters=dynamic_key[2:4], N=m * n)
    r_x3 = extract2(cip, cover, [int(m * 0.25), n], TT)
    rim = refactor(r_x3, x, y, maxx3, minx3, im_shape)

    return rim


if __name__ == '__main__':

    im = cv.imread('date/lena_gray_512.tif', 0)
    print(im.shape)
    sec_key = [0.55, 0.55, 24, 21]
    dynamic_key = im_sha256(im, sec_key)
    print(dynamic_key)
    x, y = ICM(initial=dynamic_key[0:2], parameters=dynamic_key[2:4], N=512*512)
    plt.plot(x, y, '.')

    X3, maxx3, minx3 = compression(im,x,y,T=25)
    cip, TT = embed2(X3, im, d=-5)
    r_x3 = extract2(cip, im, [int(512*0.25),512], TT)
    rim = refactor(r_x3, x, y, maxx3, minx3, im.shape)

    cv.imshow('im', im)
    cv.imshow('cip', np.uint8(cip))
    cv.imshow('rim', np.uint8(rim))
    plt.show()
    cv.waitKey(0)



