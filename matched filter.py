import math
import numpy as np
import matplotlib.mlab as mlab
import scipy as sp
from noise import *
from Ringdown_signal import *

## Matched Filtering計算関数の定義

def matched_filter(noise, signal, fs, massvector, kerrvector, tauvector, nu, thetavector):
    
    # 準備
    dt = 1.0 / fs
    L = len(noise)
    t = np.linspace(0, 1, fs)
    NFFT = fs         # Fourier変換長
    
    # PSD推定
    psd, freqs = mlab.psd(noise, Fs=fs, NFFT=NFFT)
    df = 1.0           # 積分におけるbin幅
    Lf = len(freqs)
    
    # 擬似信号作成(Fourier変換まで)
    s = noise + signal
    w = sp.signal.hann(L)           # 窓関数（ハン関数）
    w = w / np.mean(w)
    s_fft = np.fft.rfft(s * w)/L
    
    ## Matched Filter計算
    massvector = np.round(massvector, 3)
    kerrvector = np.round(kerrvector, 3)
    tauvector = np.round(tauvector, 3)
    
    ml = len(massvector)
    kl = len(kerrvector)
    tl = len(tauvector)
    thl = len(thetavector)
    
    # テンプレート作成
    templates = np.zeros((L, ml, kl, thl))

    for i in range(ml):
        for j in range(kl):
            for k in range(thl):                   
                #--------QNM parameter--------#
                f, q = QNM_parameter(kerrvector[j], massvector[i])

                f_22, f_33, f_44 = f[0], f[1], f[2]
                q_22, q_33, q_44 = q[0], q[1], q[2]
                    
                #--------信号振幅（球面調和関数）--------#
                Y22 = plus_GW_spherical_harmonics(thetavector[k], 2, 2)
                Y33 = plus_GW_spherical_harmonics(thetavector[k], 3, 3) / Y22
                Y44 = plus_GW_spherical_harmonics(thetavector[k], 4, 4) / Y22

                #--------信号振幅（ほんまもん）--------#
                A22 = 1.0                                                                             # filter側なので
                A33 = Y33 * A22 * 0.44 * (1- 4*nu)**0.45
                A44 = Y44 * A22 *(5.4 * (nu - 0.22)**2 + 0.04)
                    
                #--------テンプレート作成--------#

                temp_22 = A22 * np.exp(-math.pi * f_22 / q_22 * t) * np.sin(2 * math.pi * f_22 * t)
                temp_33 = A33 * np.exp(-math.pi * f_33 / q_33 * t) * np.sin(2 * math.pi * f_33 * t)
                temp_44 = A44 * np.exp(-math.pi * f_44 / q_44 * t) * np.sin(2 * math.pi * f_44 * t)
                    
                templates[:, i, j, k] = temp_22 + temp_33 + temp_44
                
                
    # Matched Filter
    #--------規格化-------#
    temp_fft = np.zeros((int(NFFT/2.0+1), ml, kl, thl))
    sigma_sqr = np.zeros((ml, kl, thl))
    templates_cal = np.zeros((L, ml, kl, thl))
    temp_fft_cal = np.zeros((int(NFFT/2.0+1), ml, kl, thl))
    sigma_sqr_cal = np.zeros((ml, kl, thl))

    w = sp.signal.hann(NFFT)
    w = w / np.mean(w)
    
    for i in range(ml):
        for j in range(kl):
            for k in range(thl):
                temp_fft[:, i, j, k] = np.fft.rfft(templates[:, i, j, k] * w) / NFFT
                sigma_sqr[i, j, k] = 4 * np.real(np.sum((temp_fft[1:, i, j, k] * np.conj(temp_fft[1:, i, j, k]) / psd[1:]) * df))
                templates_cal[:, i, j, k] = templates[:, i, j, k] / np.sqrt(sigma_sqr[i, j, k])
                temp_fft_cal[:, i, j, k] = np.fft.rfft(templates_cal[:, i, j, k] * w) / NFFT
                sigma_sqr_cal[i,j,k] = 4 * np.real(np.sum((temp_fft_cal[1:, i, j, k] * np.conj(temp_fft_cal[1:, i, j, k]) / psd[1:]) * df))
    
    matchedfilter_cal = np.zeros((ml, kl, tl, thl))
                
    for i in range(ml):
        for j in range(kl):
            for k in range(tl):
                for l in range(thl):
                    matchedfilter_cal[i, j, k, l] = 4 * np.real(np.sum((s_fft[1:] * np.conj(temp_fft_cal[1:, i, j, l]) * np.exp(2 * math.pi *1j * freqs[1:] * tauvector[k]) / psd[1:]) * df))
                    
    return np.abs(matchedfilter_cal)
