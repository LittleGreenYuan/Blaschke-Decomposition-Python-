# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:38:36 2021

@author: Green_yuan
"""

import numpy as np
import matplotlib as matlab
import matplotlib.pyplot as plt
from scipy.io import loadmat as load
from scipy.fftpack import fft,ifft
import math
from scipy.special import gamma
from matplotlib.colors import LogNorm
from datetime import datetime
from scipy import fftpack

#低高通滤波器，FFT遮罩实现
def getHL(f, N, over, D):
    [l,ll] = f.shape;
    ll2 = np.int(np.floor(ll/2));
    unit = np.arange(l-1,ll,1);
    unit = unit.reshape(1,len(unit))
    unit_m = np.arange(l-1,over*ll,1);
    unit_m = unit_m.reshape(1,len(unit_m))
    
    mask_N = np.zeros([l,ll*over]);
    maskLow = np.zeros([l,ll*over]);
    maskHigh = np.zeros([l,ll*over]);
    
    mask_N[0,0:N-1] = 1;
    maskLow[0,0:D] = 1;
    maskHigh[0,D:ll*over] = 1;
    
    #对信号进行过采样
    f_fft = fft(f);
    f_fft_m = np.zeros([l,ll*over])*1j;
    f_fft_m[0,0:ll2]=f_fft[0,0:ll2]*over;
    
    #过采样的信号开始高低频的分解
    Low_ana_m = ifft(f_fft_m*maskLow);
    High_ana_m = ifft(f_fft_m*maskHigh);
    
    #对信号进行降采样
    High_ana = np.zeros([l,ll])*1j;
    Low_ana = np.zeros([l,ll])*1j;
    
    temp=High_ana_m[0,np.arange(0,ll*over,over)];
    High_ana[0,0:ll] = temp.reshape(1,len(temp));
    
    temp=Low_ana_m[0,np.arange(0,ll*over,over)];
    Low_ana[0,0:ll] = temp.reshape(1,len(temp));
    return High_ana, Low_ana
#非数值法求解Blaschke乘积
def getBG(f, N, over, eps):
    [l,ll] = f.shape;
    ll2 = np.int(np.floor(ll/2));
    unit = np.arange(l-1,ll,1);
    unit = unit.reshape(1,len(unit))
    unit_m = np.arange(l-1,over*ll,1);
    unit_m = unit_m.reshape(1,len(unit_m))
    mask_N = np.zeros([l,ll*over]);
    
    mask_N[0,0:N] = 1;
    
    f_fft = fft(f)
    f_fft_m = np.zeros([l,ll*over])*1j;
    f_fft_m[0,0:ll2] = f_fft[0,0:ll2]*over;
    f_ana_m = ifft(f_fft_m)
    
    f_abs_m = np.abs(f_ana_m)
    if min(f_abs_m.T) < eps*max(f_abs_m.T):
        print('close to zero! add eps.')
        eps2 = (eps*max(f_abs_m.T))**2;
        log_abs_f = 0.5*np.log((f_abs_m**2+eps2));
    else:
        eps2 = 0;
        log_abs_f = np.log(f_abs_m);
    
    #开始进行解绕
    m = np.mean(log_abs_f);
    Ana_log_abs_F = m + 2*ifft(fft(log_abs_f-m)*mask_N);
    
    G_ana_m = ifft(fft(np.exp(Ana_log_abs_F))*mask_N);
    B_ana_m = f_ana_m/G_ana_m;
    
    
    #对信号进行降采样
    G_ana = np.zeros([l,ll])*1j;
    B_ana = np.zeros([l,ll])*1j;
    
    temp=G_ana_m[0,np.arange(0,ll*over,over)];
    G_ana[0,0:ll] = temp.reshape(1,len(temp));
    
    temp=B_ana_m[0,np.arange(0,ll*over,over)];
    B_ana[0,0:ll] = temp.reshape(1,len(temp));
    return B_ana, G_ana

#信号相位提取
def getPhase(f, N ,over, IFmethod):
    [l,ll] = f.shape;
    ll2 = np.int(np.floor(ll/2));
    unit = np.arange(l-1,ll,1);
    unit = unit.reshape(1,len(unit));
    unit_m = np.arange(l-1,over*ll,1);
    unit_m = unit_m.reshape(1,len(unit_m));
    
    f_fft = fft(f)
    f_fft_m = np.zeros([l,ll*over])*1j;
    f_fft_m[0,0:ll2] = f_fft[0,0:ll2]*over;
    
    B_ana_m = ifft(f_fft_m);
    if IFmethod == 3:
        thetaB = np.unwrap(np.angle(B_ana_m));
        [X,Y]=thetaB.shape
        der_thetaB = np.gradient(thetaB.reshape(Y,));
        der_thetaB = der_thetaB.reshape(1,len(der_thetaB))
        
        
     #对信号进行降采样
    phase_B = np.zeros([l,ll]);
    der_phase_B = np.zeros([l,ll]);
    
    temp=thetaB[0,np.arange(0,ll*over,over)];
    phase_B[0,0:ll] = temp.reshape(1,len(temp));
    
    temp=der_thetaB[0,np.arange(0,ll*over,over)];
    der_phase_B[0,0:ll] = temp.reshape(1,len(temp))*over;    
    
    return phase_B, der_phase_B





def BKdecomp(f, pass_num, N, over, D, eps, IFmethod):
    #BKD
    [l,ll]=f.shape
    if ll == l:
        raise Exception('ERROR(function_BhD 00x1):The signal must be saved as a row.')
    if np.min([l,ll])>l:
        raise Exception('ERROR(function_BhD 00x2):The code only support one channel signal right now.')

    High = np.zeros([pass_num,ll])*1j;
    Low  = np.zeros([pass_num,ll])*1j;
    G    = np.zeros([pass_num,ll])*1j;
    B    = np.zeros([pass_num,ll])*1j;
    B_prod = np.zeros([pass_num,ll])*1j;
    B_phase = np.zeros([pass_num,ll])*1j;
    B_phase_B = np.zeros([pass_num,ll])*1j;
    B_phase_der = np.zeros([pass_num,ll])*1j;
    #第一次分解
    [High_ana, Low_ana] = getHL(f, N, over, D);
    [B_ana, G_ana] = getBG(High_ana, N, over, eps);
    [phase_B, der_phase_B] = getPhase(B_ana, N ,over, IFmethod);
    
    High[0,:] = High_ana;
    Low[0,:] = Low_ana;
    G[0,:] = G_ana;
    B[0,:] = B_ana;
    B_phase[0,:] = phase_B;
    B_phase_der[0,:] = der_phase_B;
    B_prod[0,:] = B_ana;
    
    
    for k in range(1,pass_num):
        [High_ana, Low_ana] = getHL(G[0,:].reshape(1,len(G[k-1,:])), N, over, D);
        [B_ana, G_ana] = getBG(High_ana, N, over, eps);
        [phase_B, der_phase_B] = getPhase(B_ana, N, over, IFmethod);
        High[k,:] = High_ana;
        Low[k,:] = Low_ana;
        G[k,:] = G_ana;
        B[k,:] = B_ana;
        B_phase[k,:] = phase_B;
        B_phase_der[k,:] = der_phase_B;
        B_prod[k,:] = B_prod[k-1,:]*B_ana;
    
    return High, Low, B, G, B_phase,B_phase_der, B_prod
    



#信号生成
t=np.linspace(0,3,1024);
t=t.reshape(t.size,1);
#x=np.sin(25*math.pi*t*t);
y1=np.sin(2*math.pi*t*t+20*math.pi*t);
y2=np.sin(10*math.pi*t+10*math.pi*t);
y=y1+y2;
#plt.plot(t,y)#绘制x的曲线图
#spyder:Tools > Preferences > IPython Console > Graphics > Graphics backend, inline 即终端输出，Qt5则是新窗口输出。


#Blaschke解绕的参数
Cam = 1;#公式里的m
pass_num = 3;
over = 16;
D = 1;
N =20;
eps = 1e-4;
IFmethod = 3;
Alpha = 5e-5;

Carrier = 0;#载波频率,如果添加了最后需要给结果乘负载波还原

yC = y*np.exp((1j)*2*math.pi*Carrier*t);
#plt.plot(t,yC)

#镜像信号处理

mirror_yC=np.vstack((yC[np.int(len(yC)/2):np.int(len(yC))],yC));
mirror_yC=np.vstack((mirror_yC,yC[0:np.int(len(yC)/2)]));
'''
mirror_yC=np.vstack((yC,yC[::-1]));#原本代码里的对称
'''
#真正的信号被放在了中间
#true_signal=mirror_yC[np.int(len(yC)/2):np.int(len(yC)/2+len(yC))]
#值得注意的是，plt只能直接绘制（x，1）的信号
mask = np.ones([len(mirror_yC),1]);
mask[np.int(len(mask)/2):np.int(len(mask))] = 0;

fC=ifft((fft(mirror_yC.T)*mask.T))

[HighC, LowC, BC, GC, B_phaseC,B_phase_derC, B_prodC]=BKdecomp(fC, pass_num, len(fC.T), over, D, eps, IFmethod);

x0=B_prodC[0,:].reshape(1,len(B_prodC[0,:])).T;
x1=B_prodC[1,:].reshape(1,len(B_prodC[1,:])).T;
#x1=B_prodC[1,:].reshape(1,len(B_prodC[1,:])).T*LowC[0,:].reshape(1,len(LowC[0,:])).T;
x2=B_prodC[2,:].reshape(1,len(B_prodC[2,:])).T;
#x2=B_prodC[2,:].reshape(1,len(B_prodC[2,:])).T*LowC[1,:].reshape(1,len(LowC[1,:])).T;

plt.plot(y1)
plt.plot(x0[1025:1025+1024])
plt.figure()
plt.plot(y2)
plt.plot(x1[1025:1025+1024])


[phase_B, der_phase_B] = getPhase(x1.T, 2048 ,over, IFmethod);
plt.figure()
plt.plot(der_phase_B.T)
[phase_B, der_phase_B] = getPhase(x0.T, 2048 ,over, IFmethod);
plt.plot(der_phase_B.T)

