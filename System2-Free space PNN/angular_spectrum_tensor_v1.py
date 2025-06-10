# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:02:40 2021

@author: li
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
import math
from matplotlib import pyplot as plt
import cv2
Device = torch.device('cuda')  # 使用设备


def flip(img_torch):
    # 张量旋转180
    arr = torch.rot90(img_torch, dims=[2, 3])  # 要指定维度进行旋转！！！！
    arr1 = torch.rot90(arr, dims=[2, 3])
    return arr1


def resize(Ei, m):
    # 以插入相同值的方法增大矩阵维度
    # Ei.shpae=[b,c,h,w]
    # [N,M]=Ei.shape
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    Ei = torch.flatten(Ei, 2).view(b, c, N * M, 1)
    Ei = Ei.repeat(1, 1, 1, m)
    Ei = Ei.reshape(b, c, m * N * M, 1)
    Ei = Ei.repeat(1, 1, 1, m)
    Ei = Ei.reshape(b, c, N, m * M, m)
    Ei = torch.split(Ei, 1, dim=2)
    a = Ei[0].reshape(b, c, m * M, m)
    for i in range(1, N):
        a = torch.cat([a, Ei[i].reshape(b, c, m * M, m)], dim=3)
    Eo = a.transpose(2, 3)

    return Eo


def lens_phase(Ei, dx=6.3, f=150e3, n=1, lambda0=632.8e-3):
    # 透镜相位分布
    k = 2 * np.pi / lambda0 * n  # 波矢
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    x = (torch.arange(-M / 2, M / 2) * dx).to(Device)
    y = (torch.arange(-N / 2, N / 2) * dx).to(Device)
    Y, X = torch.meshgrid(x, y)
    X = X.to(Device)
    Y = Y.to(Device)
    # Eout=torch.complex(torch.exp(-k*1j*(X**2+Y**2)/(2*f))) #这里引入了复数，能行吗
    Eout_real = torch.cos(-k * (X ** 2 + Y ** 2) / (2 * f))
    Eout_imag = torch.sin(-k * (X ** 2 + Y ** 2) / (2 * f))
    Eout = torch.complex(Eout_real, Eout_imag).to(Device)
    Eoutput = Eout.view(1, 1, N, M)
    Eoutput = Eoutput.expand(b, c, -1, -1)

    return Eoutput
def lens_aperture(Ei, dx=6.3, f=150e3, n=1, lambda0=632.8e-3 ,r=12.7e3):
    # 透镜相位分布
    k = 2 * np.pi / lambda0 * n  # 波矢
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    x = (torch.arange(-M / 2, M / 2) * dx).to(Device)
    y = (torch.arange(-N / 2, N / 2) * dx).to(Device)
    Y, X = torch.meshgrid(x, y)
    X = X.to(Device)
    Y = Y.to(Device)
    # Eout=torch.complex(torch.exp(-k*1j*(X**2+Y**2)/(2*f))) #这里引入了复数，能行吗
    Eout_real = torch.cos(-k * (X ** 2 + Y ** 2) / (2 * f))
    Eout_imag = torch.sin(-k * (X ** 2 + Y ** 2) / (2 * f))
    Eout = 1*(X**2+Y**2<=r**2)+0*(X**2+Y**2>r**2)
    Eout = Eout.to(Device)
    Eoutput = Eout.view(1, 1, N, M)
    Eoutput = Eoutput.expand(b, c, -1, -1)

    return Eoutput

def lens_phase_cylindrical(Ei, dx=6.3, n=1, lambda0=632.8e-3, f=150e3):
    # 透镜相位分布
    k = 2 * np.pi / lambda0 * n  # 波矢
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    x = (torch.arange(-M / 2, M / 2) * dx).to(Device)
    y = (torch.arange(-N / 2, N / 2) * dx).to(Device)
    Y, X = torch.meshgrid(x, y)
    X = X.to(Device)
    Y = Y.to(Device)
    # Eout=torch.complex(torch.exp(-k*1j*(X**2+Y**2)/(2*f))) #这里引入了复数，能行吗
    Eout_real = torch.cos(-k * (Y ** 2) / (2 * f))
    Eout_imag = torch.sin(-k * (Y ** 2) / (2 * f))
    Eout = torch.complex(Eout_real, Eout_imag).to(Device)
    Eoutput = Eout.view(1, 1, N, M)
    Eoutput = Eoutput.expand(b, c, -1, -1)

    return Eoutput


def Diff(Ei, dx=6.3, n=1, z=150e3, lambda0=632.8e-3):
    # 基于DFFT的角谱衍射函数
    # 输出与输入维度相同
    # 参数
    k = 2 * np.pi / lambda0 * n
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    # 坐标
    Lfx = 1 / dx  # 角谱面宽度
    Lfy = Lfx
    LLfx = (torch.arange(-M / 2, M / 2) * Lfx / M).to(Device)  # 角谱单位坐标x
    LLfy = (torch.arange(-N / 2, N / 2) * Lfy / N).to(Device)  # 角谱单位坐标y
    LFY, LFX = torch.meshgrid(LLfx, LLfy)
    LFY = LFY.to(Device)
    LFX = LFX.to(Device)  # 坐标图
    H_real = torch.cos(k * z * torch.sqrt(1 - (lambda0 * LFX) ** 2 - (lambda0 * LFY) ** 2))
    H_imag = torch.sin(k * z * torch.sqrt(1 - (lambda0 * LFX) ** 2 - (lambda0 * LFY) ** 2))
    H = torch.complex(H_real, H_imag).to(Device)  # 传递函数
    H = torch.fft.fftshift(H)
    E_as = torch.fft.fft2(Ei)
    E_as_filter = torch.multiply(E_as, H)
    Eout = torch.fft.ifft2(E_as_filter)
    return Eout


def phasemask_formation(n, m):
    # 构建相位板阵列，输出维度为[1,1,N,M],M,N均需要为偶数

    Eo = torch.zeros([1, 1, n, m], dtype=torch.complex64).to(Device)

    for i in range(0, m - 1, 1):
        for j in range(0, n - 1, 1):
            if (np.mod(i, 2) == 0) and (np.mod(j, 2) == 0):
                Eo[0, 0, j, i] = 1
            elif (np.mod(i, 2) == 1) and (np.mod(j, 2) == 0):
                Eo[0, 0, j, i] = 1j
            elif (np.mod(i, 2) == 0) and (np.mod(j, 2) == 1):
                Eo[0, 0, j, i] = -1
            elif (np.mod(i, 2) == 1) and (np.mod(j, 2) == 1):
                Eo[0, 0, j, i] = -1j
        # Eo[0,0,int(2*j),int(2*i)]=1
        # Eo[0,0,int(2*j),int(2*i+1)]=1j
        # Eo[0,0,int(2*j+1),int(2*i)]=-1
        # Eo[0,0,int(2*j+1),int(2*i+1)]=-1j

    return Eo

def padding(E1, m, n):
    # 嵌到背景是m*n大小的图片中间，E1是输入图片
    b = E1.shape[0]
    c = E1.shape[1]
    N = E1.shape[2]
    M = E1.shape[3]  # 输入图片的形状
    E_b = torch.zeros([b, c, m, n], dtype=torch.complex64).to(Device)
    E_b[:, :, int(m // 2 - N // 2):int(m // 2 + N // 2), int(n // 2 - M // 2):int(n // 2 + M // 2)] = E1[:, :, :, :]
    # result = E_b[:, :, m // 2 - N // 2:m // 2 + N // 2, n // 2 - M // 2:n // 2 + M // 2]
    return E_b

def de_padding(E1, m, n):
    # 将大图E1中的m*n像素点挖出
    b = E1.shape[0]
    c = E1.shape[1]
    M = E1.shape[2]
    N = E1.shape[3]  # 输入图片的形状
    E_b = torch.zeros([b, c, m, n], dtype=torch.complex64).to(Device)
    E_b[0,0,:,:]=E1[:, :, int(M // 2 - m // 2):int(M// 2 + m // 2), int(N // 2 - n // 2):int(N // 2 + n // 2)]
    # result = E_b[:, :, m // 2 - N // 2:m // 2 + N // 2, n // 2 - M // 2:n // 2 + M // 2]
    return E_b
'''
def padding(E1, m, n):
    # 嵌到背景是m*n大小的图片中间，E1是输入图片
    b = E1.shape[0]
    c = E1.shape[1]
    N = E1.shape[2]
    M = E1.shape[3]  # 输入图片的形状
    E_b = torch.zeros([b, c, m, n], dtype=torch.complex64).to(Device)
    E_b[:, :, m // 2 - N // 2:m // 2 + N // 2, n // 2 - M // 2:n // 2 + M // 2] = E1[:, :, :, :]
    # result = E_b[:, :, m // 2 - N // 2:m // 2 + N // 2, n // 2 - M // 2:n // 2 + M // 2]
    return E_b
    '''

def filtering(m, n,M,N):

    mask = torch.zeros([1, 1, M, N], dtype=torch.complex64).to(Device)
    mask[:, :, M // 2 - m// 2:M // 2 + m // 2, N // 2 - n // 2:N // 2 + n// 2] = 1
    # result = E_b[:, :, m // 2 - N // 2:m // 2 + N // 2, n // 2 - M // 2:n // 2 + M // 2]

    return mask

def arg_embed(E1):
    b = E1.shape[0]
    c = E1.shape[1]
    N = E1.shape[2]
    M = E1.shape[3]
    result = E1[:, :, N // 2 - 56:N // 2 + 56, M // 2 - 56:M // 2 + 56]

    return result


def spectrum_m(Ei, dx=12.6, f=150e3):  # dx=6.3
    # 衍射一次得到的频谱,输出为二倍维度
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    E1 = Diff(Ei, dx, 1, f)  # 衍射1：物面到镜面1
    t1 = lens_phase(Ei, dx, f)
    # AP=torch.ones([b,c,320,320],dtype=torch.complex64).to(Device)    #孔径函数
    # AP[:,:]=1               #孔径最大

    E1t = t1 * E1
    E2 = Diff(E1t, dx, 1, f)  # 衍射2：镜面1到谱面
    E2m = modu(E2)
    # Eo=torch.abs(E2) #好像并没有用到Eo
    # return Eo,E2m
    return E2m, E2


def spectrum_m2(Ei, dx=12.6, f=150e3):  # dx=6.3
    # 衍射一次得到的频谱,输出为二倍维度
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    E1 = Diff(Ei, dx, 1, f)  # 衍射1：物面到镜面1
    t1 = lens_phase(Ei, dx, f)
    E1t = t1 * E1
    E2 = Diff(E1t, dx, 1, f)  # 衍射2：镜面1到谱面
    E2m = modu2(E2)
    # Eo=torch.abs(E2) #好像并没有用到Eo
    # return Eo,E2m
    return E2m, E2


def spectrum_real(Ei, dx=6.3, f=150e3):  # dx=6.3
    # 衍射一次得到的频谱,输出为二倍维度
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    E1 = Diff(Ei, dx, 1, f)  # 衍射1：物面到镜面1
    t1 = lens_phase(Ei, dx, f)
    E1t = E1 * t1
    Eo = Diff(E1t, dx, 1, f)  # 孔径函数
    # AP[:,:]=1               #孔径最大
    # Eo=torch.abs(E2) #好像并没有用到Eo
    # return Eo,E2m
    return Eo


def modu(Ei):
    # 构建相位板产生的相位函数，复数版

    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    Eo = torch.zeros([b, c, 2 * N, 2 * M], dtype=torch.complex64).to(Device)
    # Eo=Eo.astype(complex)
    for bb in range(b):
        for cc in range(c):

            for i in range(1, M + 1, 1):
                for j in range(1, N + 1, 1):

                    if (torch.real(Ei[bb, cc, j - 1, i - 1]) >= 0):  # 定义subpixel1
                        Eo[bb, cc, 2 * (j - 1), 2 * (i - 1)] = torch.real(Ei[bb, cc, j - 1, i - 1])
                    else:  # 定义subpixel2
                        Eo[bb, cc, 2 * (j - 1) + 1, 2 * (i - 1)] = torch.real(Ei[bb, cc, j - 1, i - 1])

                    if (torch.imag(Ei[bb, cc, j - 1, i - 1]) >= 0):  # 定义subpixel3
                        Eo[bb, cc, 2 * (j - 1), 2 * (i - 1) + 1] = 1j * torch.imag(Ei[bb, cc, j - 1, i - 1])
                    else:  # 定义subpixel4
                        Eo[bb, cc, 2 * (j - 1) + 1, 2 * (i - 1) + 1] = 1j * torch.imag(Ei[bb, cc, j - 1, i - 1])

    return Eo


def modu2(Ei):
    # 构建相位板产生的相位函数，复数版

    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    Eo = torch.zeros([b, c, 2 * N, 2 * M], dtype=torch.complex64).to(Device)
    # Eo=Eo.astype(complex)
    for bb in range(b):
        for cc in range(c):

            for i in range(1, M + 1, 1):
                for j in range(1, N + 1, 1):

                    if (torch.real(Ei[bb, cc, j - 1, i - 1]) >= 0):  # 定义subpixel1
                        Eo[bb, cc, 2 * (j - 1), 2 * (i - 1)] = torch.real(Ei[bb, cc, j - 1, i - 1])
                    else:  # 定义subpixel2
                        Eo[bb, cc, 2 * (j - 1), 2 * (i - 1) + 1] = torch.real(Ei[bb, cc, j - 1, i - 1])

                    if (torch.imag(Ei[bb, cc, j - 1, i - 1]) >= 0):  # 定义subpixel3
                        Eo[bb, cc, 2 * (j - 1) + 1, 2 * (i - 1)] = 1j * torch.imag(Ei[bb, cc, j - 1, i - 1])
                    else:  # 定义subpixel4
                        Eo[bb, cc, 2 * (j - 1) + 1, 2 * (i - 1) + 1] = 1j * torch.imag(Ei[bb, cc, j - 1, i - 1])

    return Eo


def normalization(arr):
    # 归一化
    arr1 = abs(arr)
    arr = arr / arr1.max()
    return arr


def normalization_m(arr):
    # 归一化,返回绝对值
    arr = abs(arr)
    arr = arr / arr.max()
    return arr


def Cylin_pooling(Ei, t1, f, n):
    # Ei为输入1，1，640，640的tensor，t为柱面透镜的坐标，f为焦距，n为缩小比
    # 输出为1*1*64*640 tensor
    Ei1 = Diff(Ei, 6.3, 1, f * (n + 1), 632.8e-3)
    Ei1t = torch.multiply(Ei1, t1)
    E2 = Diff(Ei1t, 6.3, 1, f * (n + 1) / n, 632.8e-3)
    Eo = E2[0, 0, 288:352, :]
    return Eo


"""12.5更新: 
将固定数据的生成（包括相位板，传递函数，镜面相位）搬运到训练之外以加速计算；
建立模拟现实相位板实现变换的4f系统模拟
"""


def Transfer_function(Ei, dx=6.3, z=150e3, n=1, lambda0=632.8e-3):
    # 通过输入面坐标和传播距离信息计算传递函数
    k = 2 * np.pi / lambda0 * n
    b = Ei.shape[0]
    c = Ei.shape[1]
    N = Ei.shape[2]
    M = Ei.shape[3]
    x = (torch.arange(-M / 2, M / 2) * dx).to(Device)
    y = (torch.arange(-N / 2, N / 2) * dx).to(Device)
    Y, X = torch.meshgrid(x, y)
    X = X.to(Device)
    Y = Y.to(Device)
    # 坐标
    Lfx = 1 / dx  # 角谱面宽度
    Lfy = Lfx
    LLfx = (torch.arange(-M / 2, M / 2) * Lfx / M).to(Device)  # 角谱单位坐标x
    LLfy = (torch.arange(-N / 2, N / 2) * Lfy / N).to(Device)  # 角谱单位坐标y
    LFY, LFX = torch.meshgrid(LLfx, LLfy)
    LFY = LFY.to(Device)
    LFX = LFX.to(Device)  # 坐标图
    H_real = torch.cos(k * z * torch.sqrt(1 - (lambda0 * LFX) ** 2 - (lambda0 * LFY) ** 2))
    H_imag = torch.sin(k * z * torch.sqrt(1 - (lambda0 * LFX) ** 2 - (lambda0 * LFY) ** 2))
    H = torch.complex(H_real, H_imag).to(Device)  # 传递函数
    H = torch.fft.fftshift(H)

    return H


def lDiff(Ei, H):
    E_as = torch.fft.fft2(Ei)
    E_as_filter = torch.multiply(E_as, H)
    Eout = torch.fft.ifft2(E_as_filter)
    return Eout


def lDiff_v(Ei, H):
    # 反向衍射传输
    H = 1 / H
    E_as = torch.fft.fft2(Ei)
    E_as_filter = torch.multiply(E_as, H)
    Eout = torch.fft.ifft2(E_as_filter)
    return Eout


def realmask_4f(Ei, T_len, TF_f, TF_D, TF_d, PM, W, f=150e3):
    # 考虑相位板实际位置的4f系统模拟
    # Ei为物面input，要求
    # T_len为镜面复透过率
    # TF_f为传播焦距距离的传播函数
    # TF_D为传播f-d距离的传播函数
    # TF_d为传播d距离的传播函数
    # f为焦距
    E1 = lDiff(Ei, TF_f)
    E1t = E1 * T_len
    E2 = lDiff(E1t, TF_D)
    E2am = E2 * PM  # 相位面1
    E3 = lDiff(E2am, TF_d)  # 谱面
    E3 = E3 * W
    E3 = lDiff(E3, TF_d)
    E3am = E3 * PM  # 相位面2
    E3t = lDiff(E3am, TF_D)
    E3t = E3t * T_len
    E4 = lDiff(E3t, TF_f)
    return E4


def phasemask_formation_double2(n, m):
    # 构建相位板阵列，输出维度为[1,1,N,M],M,N均需要为偶数

    Eo = torch.zeros([1, 1, n, m], dtype=torch.complex64).to(Device)

    for i in range(0, m, 1):
        for u in range(0, n, 1):
            if (np.mod(i, 2) == 0) and (np.mod(u, 2) == 0):
                Eo[0, 0, u, i] = 1
            elif (np.mod(i, 2) == 1) and (np.mod(u, 2) == 0):
                Eo[0, 0, u, i] = 1j
            elif (np.mod(i, 2) == 0) and (np.mod(u, 2) == 1):
                Eo[0, 0, u, i] = np.sqrt(1j)
            elif (np.mod(i, 2) == 1) and (np.mod(u, 2) == 1):
                Eo[0, 0, u, i] = np.sqrt(-1 * 1j)
        # Eo[0,0,int(2*j),int(2*i)]=1
        # Eo[0,0,int(2*j),int(2*i+1)]=1j
        # Eo[0,0,int(2*j+1),int(2*i)]=-1
        # Eo[0,0,int(2*j+1),int(2*i+1)]=-1j

    return Eo


def phasemask_formation_double1(n, m):
    # 构建相位板阵列，输出维度为[1,1,N,M],M,N均需要为偶数

    Eo = torch.zeros([1, 1, n, m], dtype=torch.complex64).to(Device)

    for i in range(0, m, 1):
        for u in range(0, n, 1):
            if (np.mod(i, 2) == 0) and (np.mod(u, 2) == 0):
                Eo[0, 0, u, i] = 1
            elif (np.mod(i, 2) == 1) and (np.mod(u, 2) == 0):
                Eo[0, 0, u, i] = np.sqrt(1j)
            elif (np.mod(i, 2) == 0) and (np.mod(u, 2) == 1):
                Eo[0, 0, u, i] = 1j
            elif (np.mod(i, 2) == 1) and (np.mod(u, 2) == 1):
                Eo[0, 0, u, i] = np.sqrt(-1 * 1j)
        # Eo[0,0,int(2*j),int(2*i)]=1
        # Eo[0,0,int(2*j),int(2*i+1)]=1j
        # Eo[0,0,int(2*j+1),int(2*i)]=-1
        # Eo[0,0,int(2*j+1),int(2*i+1)]=-1j

    return Eo


def Prop4f(Ei, T_len, TF_f, W):
    # 考虑相位板实际位置的4f系统模拟
    # Ei为物面input，要求
    # T_len为镜面复透过率
    # TF_f为传播焦距距离的传播函数
    # TF_D为传播f-d距离的传播函数
    # TF_d为传播d距离的传播函数
    # f为焦距
    E1 = lDiff(Ei, TF_f)
    E1t = E1 * T_len
    E2 = lDiff(E1t, TF_f)  # 谱面
    E2 = E2 * W
    E3 = lDiff(E2, TF_f)
    E3 = E3 * T_len
    E4 = lDiff(E3, TF_f)
    return E4


def realmask_4f_singleword(Ei, T_len, TF_f, TF_D, TF_d, PM, W, f=150e3):
    # 考虑相位板实际位置的4f系统模拟
    # Ei为物面input，要求
    # T_len为镜面复透过率
    # TF_f为传播焦距距离的传播函数
    # TF_D为传播f-d距离的传播函数
    # TF_d为传播d距离的传播函数
    # f为焦距
    E1 = lDiff(Ei, TF_f)
    E1t = E1 * T_len
    E2 = lDiff(E1t, TF_f)  # 谱面
    E2 = E2 * W
    E3 = lDiff(E2, TF_f)
    E3 = E3 * T_len
    E3t = lDiff(E3, TF_d)
    E4 = lDiff(E3t, TF_D)
    return E4


def realmask_4f_singleword(Ei, T_len, TF_f, TF_D, TF_d, PM, W, f=150e3):
    # 考虑相位板实际位置的4f系统模拟
    # Ei为物面input，要求
    # T_len为镜面复透过率
    # TF_f为传播焦距距离的传播函数
    # TF_D为传播f-d距离的传播函数
    # TF_d为传播d距离的传播函数
    # f为焦距
    E1 = lDiff(Ei, TF_f)
    E1t = E1 * T_len
    E2 = lDiff(E1t, TF_f)  # 谱面
    E2 = E2 * W
    E3 = lDiff(E2, TF_f)
    E3 = E3 * T_len
    E3t = lDiff(E3, TF_d)
    E4 = lDiff(E3t, TF_D)
    return E4


'''12.9g更新：计算频谱在相位板有一定距离时的等效函数 结果：大G特G'''


def spectrum_reshape(Em, T_len, TF_f, TF_d, PM, f=150e3):
    # 考虑相位板实际位置的4f系统模拟
    # Ei为物面input，要求
    # T_len为镜面复透过率
    # TF_f为传播焦距距离的传播函数
    # TF_d为传播d距离的传播函数
    # f为焦距
    PM2 = PM ** 2
    E1 = lDiff(Em * PM2, TF_f)
    E1t = E1 * T_len
    E2 = lDiff(E1t, TF_d)  # 谱面
    E2t = E2 / PM
    E1t_v = lDiff_v(E2t, TF_d)
    E1t_vt = E1t_v / T_len
    Emo = lDiff_v(E1t_vt, TF_f)
    return Emo


def spectrum_reshape2(Em, T_len, TF_f, TF_d, PM, f=150e3):
    # 考虑相位板实际位置的4f系统模拟
    # Ei为物面input，要求
    # T_len为镜面复透过率
    # TF_f为传播焦距距离的传播函数
    # TF_d为传播d距离的传播函数
    # f为焦距
    PM2 = PM ** 2
    E1 = lDiff(Em * PM2, TF_d)
    E1t = E1 / PM2
    Emo = lDiff_v(E1t, TF_d)
    return Emo, E1


def make_pat_func(forward_f, backward_f):
     '''
     A function that constructs and returns the custom autograd function for physics−aware training.
     Parameters
     −−−−−−−−−−
     f forward: The function that is applied in the forward pass.
     Typically, the computation is performed on a physical system that is connected and controlled by
     the computer that performs the training loop.
     It takes in an input PyTorch tensor x, a parameter PyTorch tensor theta and returns the output PyTorch tensor y.
     f backward: The function that is employed in the backward pass to propagate estimators of gradients.
     It takes in an input PyTorch tensor x, an input PyTorch theta and returns the output PyTorch tensor y.
     Returns
     −−−−−−−
     f pat: The custom autograd function for physics−aware training.
     It takes in an input PyTorch tensor x, an input PyTorch theta and returns the output PyTorch tensor y.
     '''
     class func(torch.autograd.Function):
         @staticmethod
         def forward(ctx, x, theta): #here ctx is an object that stores data for the backward computation.
             ctx.save_for_backward(x, theta) #ctx is used to save the input and parameter of this function.
             return forward_f(x, theta)

         @staticmethod
         def backward(ctx, grad_output):
             x, theta = ctx.saved_tensors #load the input and parameters that are stored in the forward pass
             torch.set_grad_enabled(True) #autograd has to be enabled to perform jacobian computation.

             # Perform vector jacobian product of the backward function with PyTorch.
             y = torch.autograd.functional.vjp(backward_f, (x, theta), v=grad_output)
             torch.set_grad_enabled(False) #autograd should be restored to off state after jacobian computation.
             return y[1]
     f_pat = func.apply
     return f_pat


""" update 0509 2024 """


def preprocess(E, M, N, extend):
    """ preprocess 8 bit image into [a,b,M,N] tensor and extended it to [a,b,extend,extend] """
    Ei = np.zeros([1, 1, N, M])
    Ere = cv2.resize(E, [M, N])

    Ei[0, 0, :, :] = Ere

    Ei = torch.from_numpy(Ei).to('cuda')
    Ei2 = padding(Ei, extend, extend)
    Ei2 - torch.abs(Ei2)
    return Ei2
def plot_tensor(E_output, M,N):
    """ show the [a,b,c,d] tensor as  M*N 2d matrix"""
    E_output = de_padding(E_output, M, N)
    Eout = E_output[0, 0, :, :].detach().to('cpu')
    Eout=255*torch.abs(Eout)
    plt.imshow(Eout, cmap='gray')

    return


def IPSF(dx, L1, f, R, L2, M=256, N=256, extend=4096, window=11, th=0.001, lambda_0=0.532):
    ''' calculate the incoherent Point Spread Function (PSF) of an equivalent single lens system '''
    E_in = np.zeros([M, N])
    E_in[M // 2, N // 2] = 255
    E_in_t = preprocess(E_in, M, N, extend).to(Device)

    """lens function"""
    phase_lens = lens_phase(E_in_t, dx, f, 1, lambda_0)
    aperture_lens = lens_aperture(E_in_t, dx, f, 1, lambda_0, r=R)
    lens = phase_lens * aperture_lens
    lens.to(Device)
    T1=Transfer_function(E_in_t, dx, z=L1, n=1, lambda0=lambda_0).to(Device)
    L2.to(Device)

    T2=Transfer_function(E_in_t, dx, z=L2, n=1, lambda0=lambda_0).to(Device)

    """propagation"""
    E_out1 = lDiff(E_in_t,T1)
    E_lens = (E_out1 * lens).to(Device)
    E_out2 = lDiff(E_lens, T2)

    """fourier spectrum"""
    E_out_F = torch.fft.fftshift(torch.fft.fft2(E_out2))
    """postprocess"""
    t = torch.abs(E_out2)
    t = t / torch.max(torch.abs(t))
    t = t * torch.conj(t)
    window = (window - 1) // 2
    t = t[:, :, extend // 2 - window:extend // 2 + 1 + window, extend // 2 - window:extend // 2 + 1 + window]
    t_0 = torch.where(t < th, torch.zeros_like(t), t).to(Device)

    return t_0, E_out_F



def IncoDiff_MC(E_in,dx,L1,f,R,L2,M,N,extend,lambda_0=0.532,num_iter=1000):
    ''' calculate the incoherent propagation by Monter-Calro iterations'''
    E_in_t=E_in
    """lens definition"""
    phase_lens = lens_phase(E_in_t, dx, f, 1, lambda_0)
    aperture_lens = lens_aperture(E_in_t, dx, f, 1, lambda_0, r=R)
    lens = phase_lens * aperture_lens
    E_out_ave=torch.zeros([num_iter,1,extend,extend])
    E_out_final=torch.zeros([1,1,extend,extend])
    for i in range(num_iter):
        '''random phase initialization'''
        p=torch.rand(1,1,extend,extend)
        p=torch.exp(1j*2*torch.pi*p).to('cuda')
        E_in_t=E_in_t*p
        '''propagation'''
        E_out1 = lDiff(E_in_t, Transfer_function(E_in_t, dx, z= L1, n=1, lambda0=lambda_0))
        E_lens=E_out1*lens
        E_out2=lDiff(E_lens, Transfer_function(E_in_t, dx, z= L2, n=1, lambda0=lambda_0))
        '''postprocessing'''
        t=E_out2/torch.max(torch.abs(E_out2))
        E_out_ave[i-1,:,:,:]=t

    E_II_iter=torch.mean(torch.abs(E_out_ave),dim=0)
    E_II_iter=torch.abs(E_II_iter)/torch.max(torch.abs(E_II_iter))
    E_out_final[0,:,:,:]=E_II_iter
    return E_out_final

def InCo_E2E(E_in,PSF):
    E_out=torch.sqrt(torch.conv2d(torch.abs(E_in)**2,PSF,padding='same'))
    return E_out
def InCo_I2I(I_in,PSF):
    I_out=torch.conv2d(I_in,PSF,padding='same')
    return I_out

def InCo_I2I_2D(I_in,PSF):
    I_out=torch.conv2d(I_in,PSF.squeeze(0).squeeze(0),padding='same')
    return I_out

"""update 0510 2024"""
def BigPicture_single_tensor(tensor_in, mask, RawNumber=12, ColumnNumber=12, paddcol=34, paddraw=40,dim_in=28,num_out=144):
    """iamge && weight mapping for incoheretn system"""
    padd1 = paddcol
    padd2 = paddraw
    BigPictureRaw = RawNumber * padd2
    BigPictureColumn = ColumnNumber * padd1
    b = tensor_in.shape[0]
    c = tensor_in.shape[1]
    BigPicture = torch.zeros([b, c, BigPictureRaw, BigPictureColumn])
    p1 = (padd1 - dim_in) // 2
    p2 = (padd2 - dim_in) // 2
    for i in range(RawNumber):
        for j in range(ColumnNumber):
            if ((i*ColumnNumber+j)<num_out):
                im = torch.reshape(tensor_in[:, :, :, i * ColumnNumber + j], [b, c, dim_in, dim_in])
                BigPicture[:, :, i * padd2:(i + 1) * padd2, j * padd1:(j + 1) * padd1] = F.pad(im, (p1, p1, p2, p2))
            else:
                break

    Big_re = torch.repeat_interleave(BigPicture, 2, dim=3).to(Device)
    Big_re = Big_re * mask
    Big_re = Big_re.to(Device)
    return Big_re


def create_Binary_mask(n, m):
    mask_col = torch.tensor([1 if i%2==0 else 0 for i in range(m)])

    mask = mask_col.expand(n, -1)

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask= mask.to(Device)

    return mask

def create_unit_mask(n, m):
    mask_col = torch.ones([n,m])

    mask = mask_col.expand(n, -1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask= mask.to(Device)

    return mask


#%%
def SUBBLOCK_sum(tensor_in, RawNumber=12, ColumnNumber=12, padd2=40, padd1=34):
    """iamge && weight mapping for incoheretn system"""
    Tensor_size=RawNumber*ColumnNumber
    b=tensor_in.shape[0]
    c=tensor_in.shape[1]
    tensor_out=torch.zeros([b,c,Tensor_size])

    for i in range(RawNumber):
        for j in range(ColumnNumber):

            tensor_out[:,:,j+i*ColumnNumber]=tensor_in[:, :, i * padd2:(i + 1) * padd2, j * padd1*2:(j + 1) * padd1*2].sum(dim=(2, 3))
    tensor_out=tensor_out.unsqueeze(3)

    return tensor_out


def BigPicture2Vector(tensor, RawNumber=12, ColumnNumber=12,  paddraw=40,paddcol=34,dim_in=28,num_out=144):
    """iamge && weight mapping for incoheretn system"""

    b = tensor.shape[0]
    c = tensor.shape[1]
    padd1 = paddcol
    padd2 = paddraw
    tensor_out = torch.zeros([b, c, dim_in*dim_in,ColumnNumber*RawNumber])
    tensor_in = tensor[:, :, :, ::2]

    p1 = (padd1 - dim_in) // 2
    p2 = (padd2 - dim_in) // 2
    for i in range(RawNumber):
        for j in range(ColumnNumber):
            if ((i*ColumnNumber+j)<num_out):
                tensor_out[:,:,:,i*ColumnNumber+j]=torch.reshape(tensor_in[:, :, p2+i * padd2:p2+i*padd2+dim_in, p1+j * padd1:p1+j *padd1+ dim_in],[b,c,dim_in*dim_in])
            else:
                break

    tensor_out.to(Device)
    return tensor_out


def BigPicture_single_tensor_2D(tensor_in, mask, RawNumber=12, ColumnNumber=12, paddcol=30, paddraw=32,dim_in=28,num_out=144):
    """iamge && weight mapping for incoheretn system"""
    padd1 = paddcol
    padd2 = paddraw

    BigPictureRaw = RawNumber * padd2
    BigPictureColumn = ColumnNumber * padd1
    BigPicture = torch.zeros([BigPictureRaw, BigPictureColumn])
    p1 = (padd1 - dim_in) // 2
    p2 = (padd2 - dim_in) // 2
    for i in range(RawNumber):
        for j in range(ColumnNumber):
            if ((i*ColumnNumber+j)<num_out):
                im = torch.reshape(tensor_in[:, i * ColumnNumber + j], [dim_in, dim_in])
                BigPicture[ i * padd2:(i + 1) * padd2, j * padd1:(j + 1) * padd1] = F.pad(im, (p1, p1, p2, p2))
            else:
                break

    Big_re = torch.repeat_interleave(BigPicture, 2, dim=1).to(Device)

    Big_re = Big_re * mask
    Big_re = Big_re.to(Device)
    return Big_re


def BigPicture2Vector_2D(tensor, RawNumber=12, ColumnNumber=12,  paddraw=30,paddcol=32,dim_in=28, num_out=144):
    """iamge && weight mapping for incoheretn system"""

    padd1 = paddcol
    padd2 = paddraw
    tensor_out = torch.zeros([dim_in*dim_in,num_out])
    tensor_in=tensor[0,:,::2]

    p1 = (padd1 - dim_in) // 2
    p2 = (padd2 - dim_in) // 2
    for i in range(RawNumber):
        for j in range(ColumnNumber):
            if ((i*ColumnNumber+j)<num_out):
                tensor_out[:,i*ColumnNumber+j]=torch.reshape(tensor_in[p2+i * padd2:p2+i*padd2+dim_in, p1+j * padd1:p1+j *padd1+ dim_in],[dim_in*dim_in])
            else:
                break
    tensor_out.to(Device)
    return tensor_out


def create_Binary_mask_2D(n, m):
    mask_col = torch.tensor([1 if i%2==0 else 0 for i in range(m)])

    mask = mask_col.expand(n, -1)

    mask= mask.to(Device)

    return mask


def create_unit_mask_2D(n, m):
    mask_col = torch.ones([n,m])

    mask = mask_col.expand(n, -1)

    mask= mask.to(Device)

    return mask


def plot_tensor_2D(E_output):
    """ show the [c,d] tensor as  M*N 2d matrix"""
    Eout = E_output[ :, :].detach().to('cpu')
    Eout=255*torch.abs(Eout)
    plt.imshow(Eout, cmap='gray')

    return

"""updated at 2024 0528"""
def MiddlePicture_single_tensor(tensor_in,mask, RawNumber= 3 , ColumnNumber= 4 , paddcol=30, paddraw=32,dim_in=28):
    """iamge && weight mapping for incoheretn system"""
    BigPictureRaw = RawNumber * paddraw
    BigPictureColumn = ColumnNumber * paddcol * 2
    b=np.ceil(tensor_in.shape[1]/(RawNumber*ColumnNumber)).astype(np.uint8)
    N_onepic=RawNumber * ColumnNumber
    c = 1
    BigPicture = torch.zeros([b, c, BigPictureRaw, BigPictureColumn])

    for i in range(b):
        BigPicture[i,0,:,:] = BigPicture_single_tensor_2D(tensor_in[:,N_onepic*i:N_onepic*(i+1)],mask,RawNumber=RawNumber,ColumnNumber=ColumnNumber,paddcol=paddcol,paddraw=paddraw,dim_in=dim_in)



    Big_re = BigPicture.to(Device)
    return Big_re


def MiddlePicture2Vector(tensor, RawNumber=3, ColumnNumber=4,  paddcol=30,paddraw=32,dim_in=28):
    """image && weight mapping for incoherent system"""

    N = tensor.shape[0]


    tensor_out = torch.zeros([dim_in*dim_in,ColumnNumber*RawNumber*N])


    N_onepic = RawNumber * ColumnNumber

    for i in range(N):
        tensor_out[:,N_onepic*i:N_onepic*(i+1)]=BigPicture2Vector_2D(tensor[i,:,:,:],ColumnNumber=ColumnNumber,RawNumber=RawNumber,paddcol=paddcol,paddraw = paddraw,num_out=12)

    tensor_out.to(Device)
    return tensor_out