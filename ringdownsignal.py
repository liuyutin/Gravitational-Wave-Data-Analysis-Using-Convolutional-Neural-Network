# リングダウン重力波信号生成に用いる関数コード

'''

Ringdown_signal_generator : リングダウン信号生成関数
    input : ２天体の質量、スピン（kerr）、時間ベクトル、基本波の振幅、Inclination、GW到達時間のindex
    output : signalベクトル

QNM_parameter : QNM波の周波数、Q値を計算する関数
    input : スピン、２天体合体後の質量
    output : f(配列)、q(配列)

plus_GW_spherical_harmonics : 球面調和関数の振幅計算関数
    input : Inclination、モード２つ
    output : 振幅

spinweightedsphericalharmonics : 球面調和関数の振幅計算関数(sYlmのやつ)
    input : Inclination、位相、インデックス
    output : sYlm

small_dmatrix : Wignerのsmall D行列の計算関数
    input : Inclination、インデックス
    output : Wignerのsmall D行列

'''

import math
import numpy as np



def Ringdown_signal_generator(m1, m2, a, t, A22, theta, tau_index):    # a:spin, theta:inclination, tau_index:GW到達時間のindex
    q = m2 / m1
    if q < 1:
        q = 1 / q
        
    nu = q / ((1 + q)**2)                         # mass ratio
    
    m = round((m1 + m2) * 0.96, 1)     # merger後の質量：4%がGWエネルギーに消えていくので
    a = round(a, 3)
    
    t[:tau_index] = 0
    t[tau_index:] = t[tau_index:] - t[tau_index]      # tの整形

    # モードごとの周波数とQ値
    f, q = QNM_parameter(a, m)
    
    f_22 = f[0]
    f_33 = f[1]
    f_44 = f[2]
    q_22 = q[0]
    q_33 = q[1]
    q_44 = q[2]
    
    # 信号振幅（球面調和関数）
    Y22 = plus_GW_spherical_harmonics(theta, 2, 2)
    Y33 = plus_GW_spherical_harmonics(theta, 3, 3) / Y22
    Y44 = plus_GW_spherical_harmonics(theta, 4, 4) / Y22
    
    # 信号振幅（ほんまもん）
    A33 = Y33 * A22 * 0.44 * (1- 4*nu)**0.45
    A44 = Y44 * A22 *(5.4 * (nu - 0.22)**2 + 0.04)
    
    # signal生成
    
    signal_22 = A22 * np.exp(-math.pi * f_22 / q_22 * t) * np.sin(2 * math.pi * f_22 * t)
    signal_33 = A33 * np.exp(-math.pi * f_33 / q_33 * t) * np.sin(2 * math.pi * f_33 * t)
    signal_44 = A44 * np.exp(-math.pi * f_44 / q_44 * t) * np.sin(2 * math.pi * f_44 * t)
    signal = signal_22 + signal_33 + signal_44
    
    return np.real(signal)
    
def QNM_parameter(a,m):
    f_sig = np.array([[1.5251, -1.1568, 0.1292], [1.8956, -1.3043, 0.1818], [2.3000, -1.5056, 0.2244]])     # 22, 33, 44モードのみ、行がモード、列がインデックス
    q_sig = np.array([[0.7000, 1.4187, -0.4990], [0.9000, 2.3430, -0.4810], [1.1929, 3.1191, -0.4825]])
    
    f = np.zeros(3)
    q = np.zeros(3)
    
    for i in range(len(f_sig)):
        f[i] = 32 * (f_sig[i, 0] + f_sig[i, 1] * (1 - a)**f_sig[i, 2]) / m * 10**3
        q[i] = q_sig[i, 0] + q_sig[i, 1] * (1 - a)**q_sig[i, 2]
        
    return f, q

def plus_GW_spherical_harmonics(theta, l, m):       # theta:Inclination, l,m:モード
    s = -2
    phi = 0         # 定義にそう書いてあったらしい
    
    y1 = spinweightedsphericalharmonics(theta, phi, l, m, s)
    y2 = (-1)**(-1 * s + m) * np.conj(spinweightedsphericalharmonics(theta, phi, l, m, s))
    
    Ylm = y1 + (-1)**l * y2
    
    return Ylm

def spinweightedsphericalharmonics(theta, phi, l, m, s):
    n = -1 * s
    d = small_dmatrix(theta, l, m, n)
    
    sYlm = (-1)**s * np.sqrt((2*l + 1) / 4 * math.pi) * d * np.exp(1j * m * phi)
    
    return sYlm

def small_dmatrix(theta, l, m, n):
    maxk = min(l - m, l - n) + 1
    SUM = 0
    
    for k in range(maxk):
        
        SUM += (-1)**k * (np.sin(theta / 2))**(2*l - m - n - 2*k) * (np.cos(theta / 2))**(m + n + 2*k) / (math.factorial(k) * math.factorial(l - m - k) * math.factorial(l - n - k) * math.factorial(m + n + k))
        
    d = (-1)**(l - n) * np.sqrt(math.factorial(l + m) * math.factorial(l - m) * math.factorial(l + n) * math.factorial(l - n)) * SUM
    
    return d
