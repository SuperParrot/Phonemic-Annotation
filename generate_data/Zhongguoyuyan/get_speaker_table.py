import os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy.io as scio
import traceback


if __name__=='__main__':
    source_root_paths = ['F:/Datasets/Zhongguoyuyan/train/', 'F:/Datasets/Zhongguoyuyan/dev/', 'F:/Datasets/ChineseMarking/raw/train/']
    save_path = 'F:/Datasets/Zhongguoyuyan/'

    cnt=0
    speaker_table=[]
    for source_root_path in source_root_paths:
        for filepath, dirnames, filenames in os.walk(source_root_path):
            for filename in filenames:
                # print(filename)

                if (filename == 'label.xlsx'):
                    speaker='_'.join(filepath.replace('\\','/').split('/')[-3:])
                    if(not speaker in speaker_table):
                        speaker_table.append(speaker)

                    cnt+=1

    print('%d file(s) has been found.' % cnt)
    print('Number of speakers: %d'%(len(speaker_table)))

    print(speaker_table)
    f_write=open(save_path+'speaker_table.txt', 'w', encoding='utf-8')
    f_write.write('\n'.join(speaker_table))
    f_write.close()
