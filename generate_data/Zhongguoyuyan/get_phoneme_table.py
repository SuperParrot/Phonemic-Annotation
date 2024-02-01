import os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy.io as scio
import traceback


if __name__=='__main__':
    #source_root_paths = ['F:/Datasets/Zhongguoyuyan/train/', 'F:/Datasets/Zhongguoyuyan/dev/', 'F:/Datasets/ChineseMarking/raw/train/']
    #save_path = 'F:/Datasets/Zhongguoyuyan/'

    source_root_paths = ['F:/Datasets/Zhongguoyuyan/dev/raw/Xian/']
    save_path = './'

    cnt=0
    phoneme_table=[]
    for source_root_path in source_root_paths:
        for filepath, dirnames, filenames in os.walk(source_root_path):
            for filename in filenames:
                # print(filename)

                if (filename == 'label.xlsx'):
                    label_data = pd.read_excel(filepath + '/' + 'label.xlsx', sheet_name='Sheet1', header=0)

                    for i in range(len(label_data)):
                        sylla_num = label_data['num_of_syllables'][i]
                        for j in range(2 * sylla_num):
                            phonemes=str(label_data['phoneme_' + str(j)][i])
                            phonemes=phonemes.replace(' ', '')

                            for phoneme in phonemes:
                                if((not phoneme in phoneme_table) and phoneme!='-'):
                                    phoneme_table.append(phoneme)
                    #print(phoneme_table)
                    #exit(0)

                    cnt+=1

    print('%d file(s) has been found.' % cnt)
    print('Number of phonemes: %d'%(len(phoneme_table)))

    print(phoneme_table)
    f_write=open(save_path+'phoneme_table.txt', 'w', encoding='utf-8')
    f_write.write('\n'.join(phoneme_table))
    f_write.close()
