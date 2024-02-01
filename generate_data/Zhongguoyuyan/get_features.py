import os
import numpy as np
import pandas as pd
from moviepy.editor import *
import librosa as librosa
import librosa.display
import pygame
import scipy.signal as signal
import scipy.io as scio
import matplotlib.pyplot as plt
import traceback
import soundfile
import random

'''
————————————————
版权声明：本文为CSDN博主「吃块小西瓜」的原创文章，遵循CC 4.0 BY - SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / weixin_45272908 / article / details / 115641702
'''

def read_wav(path, sample_rate):
    wav_dura = librosa.get_duration(filename=path)
    y, sr = librosa.load(path, sr=sample_rate, offset=0.0, duration=wav_dura)
    '''
    x = np.arange(0, wav_dura, 1 / sr)  # 时间刻度
    plt.plot(x, y)
    plt.xlabel('times')  # x轴时间
    plt.ylabel('amplitude')  # y轴振幅
    plt.title(f, fontsize=12, color='black')  # 标题名称、字体大小、颜色
    plt.show()
    '''
    return wav_dura, y

def remove_blanks(sig, lframe):
    result=[]
    start=0
    while(start<len(sig)):
        frame=sig[start:min(start+lframe, len(sig))]
        #print(frame.shape)
        #print(np.abs(frame).mean())
        if(np.abs(frame).mean()>1e-3):
            result.extend(sig[start:min(start+lframe, len(sig))])

        start+=lframe

    result=np.array(result)
    return result

#Pre-emphasis
def preemphasis(audio_sig):
    signal_points = len(audio_sig)  # 获取语音信号的长度
    signal_points = int(signal_points)  # 把语音信号的长度转换为整型
    # s=x  # 把采样数组赋值给函数s方便下边计算
    for i in range(1, signal_points, 1):  # 对采样数组进行for循环计算
        audio_sig[i] = audio_sig[i] - 0.98 * audio_sig[i - 1]  # 一阶FIR滤波器

    '''
    x = np.arange(0, wav_dura, 1 / 24000)
    plt.plot(x, audio_sig)  # 绘出图形
    plt.xlabel('times')  # x轴时间
    plt.ylabel('amplitude')  # y轴振幅
    plt.title('pre_emphasis', fontsize=12, color='black')  # 标题名称、字体大小、颜色
    plt.show()
    '''

    return audio_sig  # 返回预加重以后的采样数组


if __name__=='__main__':
    seed = 114514
    random.seed(seed)

    skip = True

    #source_root_path = 'F:/Datasets/Zhongguoyuyan/train/raw/'
    source_root_path = 'F:/Datasets/Zhongguoyuyan/dev/raw/'
    data_aug = False

    sample_rate = 22050
    lframe = int(sample_rate * 0.025)  # 帧长(持续0.025秒)
    mframe = int(sample_rate * 0.001)  # 帧移

    cnt = 0

    for filepath, dirnames, filenames in os.walk(source_root_path):
        for dirname in dirnames:
            # print(os.path.join(filepath, dirname))
            for filename in filenames:
                # print(filename)

                if (dirname == 'Video' and filename == 'label.xlsx'):
                    file_list = os.listdir(filepath + '/' + dirname)

                    for mp4FileName in file_list:
                        if (not mp4FileName.endswith('.mp4')):
                            continue

                        save_path = filepath + '/features/'
                        if (not os.path.exists(save_path)):
                            os.mkdir(save_path)
                        save_name = save_path + mp4FileName.replace('.mp4', '_features.mat')
                        if (skip and os.path.exists(save_name)):
                            continue

                        video = VideoFileClip(filepath + '/' + dirname + '/' + mp4FileName)
                        audio = video.audio
                        audio.write_audiofile('./tmp.wav')

                        wav_dura, audio_sig = read_wav('./tmp.wav', sample_rate)
                        removed=False
                        while not removed:
                            try:
                                os.remove('./tmp.wav')
                                removed = True
                            except:
                                pass

                        audio_sig = audio_sig / audio_sig.max()

                        print(mp4FileName)
                        '''
                        print(audio_sig.shape)
                        plt.plot(audio_sig)
                        plt.show()
                        '''
                        '''
                        plt.subplot(1,2,1)
                        plt.plot(audio_sig)
                        '''
                        audio_sig = remove_blanks(audio_sig, 240)
                        '''
                        plt.subplot(1, 2, 2)
                        plt.plot(audio_sig)
                        plt.show()
                        '''

                        #soundfile.write('./tmp1.wav', audio_sig, 22050)
                        #exit(0)


                        '''
                        print(audio_sig.shape)
                        plt.plot(audio_sig)
                        plt.show()
                        scio.wavfile.write('./tmp.wav', sample_rate, audio_sig)
                        '''

                        audio_sig = preemphasis(audio_sig)

                        # MFCC
                        # print(audio_sig.shape)
                        mfccs = librosa.feature.mfcc(y=audio_sig, sr=sample_rate, n_mfcc=64)
                        #print(mfccs.shape)
                        '''
                        #audio=librosa.feature.inverse.mfcc_to_audio(mfccs, sr=22050, n_mels=64)
                        #soundfile.write('./tmp1.wav', audio, 22050)
                        #exit(0)
                        '''

                        scio.savemat(save_name, {'features': mfccs})

                        if data_aug:
                            # print(x_wave.shape)
                            for i in range(5):
                                #print(random.random())

                                # Time stretch
                                x_wave = librosa.effects.time_stretch(audio_sig, rate= 1.5 * random.random() + 0.5)
                                mfccs = librosa.feature.mfcc(y=x_wave, sr=sample_rate, n_mfcc=64)
                                scio.savemat(save_name.replace('_features.mat', 'timeAug'+str(i)+'_features.mat'), {'features': mfccs})

                                # Pitch stretch
                                x_wave = librosa.effects.pitch_shift(audio_sig, sr=sample_rate,
                                                                     n_steps=10.0 * random.random() - 5.0)
                                mfccs = librosa.feature.mfcc(y=x_wave, sr=sample_rate, n_mfcc=64)
                                scio.savemat(save_name.replace('_features.mat', 'pitchAug'+str(i)+'_features.mat'), {'features': mfccs})

                        cnt += 1

                        #exit(0)

    print('%d file(s) has been processed.'%cnt)
