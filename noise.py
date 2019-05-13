# GW150914の生データから、パーミュテーションテスト後のノイズを生成するための関数


'''

data_format : txtデータからnp配列に変換する関数
    input : txtデータの名前
    output : np配列データ

rand_sample : パーミュテーション後のノイズを生成する関数
    input : 整形されたデータ、サンプリング周波数、データ長
    output : パーミュテーションテスト後のノイズ


'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



def data_format(data_name):

    with open(data_name, 'r') as f:
        data = f.read()

    data = data.split('\n')
    del data[-1]

    for i in range(len(data)):
        data[i] = float(data[i])

    data = np.array(data)

    return data


def rand_sample(data, fs, L):
    # PSD推定
    dt = 1.0 / fs
    NFFT = fs
    
    psd, freqs = mlab.psd(data, Fs=fs, NFFT=NFFT)
    
    # ホワイトニング
    Nt = L
    data2 = data
    
    hf = np.fft.rfft(data2)
    white_hf = hf / np.sqrt(psd / dt / 2)
    white_hf = np.fft.irfft(white_hf, Nt)
    
    # ランダム化
    rand_hf = np.random.permutation(white_hf)
    rand_hf = np.fft.rfft(rand_hf)
    rand_hf = rand_hf * np.sqrt(psd / dt / 2)
    rand_hf = np.fft.irfft(rand_hf)
    
    norm_factor = np.median(np.abs(rand_hf)) / np.median(np.abs(data))
    rand_hf = rand_hf / norm_factor
    
    return rand_hf


