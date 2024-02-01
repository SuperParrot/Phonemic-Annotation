import os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy.io as scio
import traceback

def find_in_label(label_data, word_name):
    #print(word_name)
    for i in range(5):
        word_name=word_name.replace('timeAug'+str(i),'')
        word_name=word_name.replace('pitchAug'+str(i),'')
    #print(word_name)

    for i in range(len(label_data)):
        #print(label_data['Word'][i])
        if(label_data['Word'][i] == word_name):
            return i

    return -1

def get_phoneme_names(table_paths):
    phoneme_names=[]

    for table_path in table_paths:
        f_read = open(table_path + '/phoneme_table.txt', encoding='utf-8')
        for line in f_read:
            phoneme_read = line.strip()
            if (not phoneme_read in phoneme_names):
                phoneme_names.append(phoneme_read)
        f_read.close()

    return phoneme_names

if __name__=='__main__':
    skip = False

    source_root_path = 'F:/Datasets/Zhongguoyuyan/train/raw/'
    save_path = 'F:/Datasets/Zhongguoyuyan/train/generated/'

    #source_root_path = 'F:/Datasets/Zhongguoyuyan/dev/raw/'
    #save_path = 'F:/Datasets/Zhongguoyuyan/dev/generated/'

    sample_rate=22050

    cnt=0

    label_phoneme_names = ['?','-']
    label_phoneme_names.extend(get_phoneme_names(['F:/Datasets/Zhongguoyuyan/']))
    print('Number of phonemes: %d' % len(label_phoneme_names))

    for filepath, dirnames, filenames in os.walk(source_root_path):
        for dirname in dirnames:
            # print(os.path.join(filepath, dirname))
            for filename in filenames:
                # print(filename)

                if (dirname == 'features' and filename == 'label.xlsx'):
                    label_data = pd.read_excel(filepath + '/' + 'label.xlsx', sheet_name='Sheet1', header=0)
                    #print(label_data['Word'])

                    save_subpath = '/' + '/'.join(filepath.replace('\\', '/').split('/')[-3:]) + '/'
                    #print(save_subpath)
                    if (not os.path.exists(save_path + save_subpath)):
                        os.makedirs(save_path + save_subpath)
                    #print(save_path+save_subpath)
                    #print(filepath)

                    for file_name in os.listdir(filepath + '/' + dirname):
                        if (not file_name.endswith('.mat')):
                            continue
                        #print(file_name)

                        save_name=save_path+save_subpath+file_name.replace('_features','_featuresWithLabel')
                        #print(save_name)

                        if (skip and os.path.exists(save_name)):
                            continue

                        print('Processing %s...' % file_name)

                        word_name=file_name.split('_')[0]
                        idx=find_in_label(label_data, word_name)
                        if(idx<0):
                            print('%s not found in label data.'%file_name)
                            continue

                        word=label_data['Word'][idx]
                        sylla_num=label_data['num_of_syllables'][idx]
                        #print(word)

                        label_phonemes=[]
                        for i in range(2*sylla_num):
                            phonemes_name=str(label_data['phoneme_'+str(i)][idx]).strip()
                            if (phonemes_name == '-'):
                                continue
                            for phoneme_name in phonemes_name:
                                label_phonemes.append(label_phoneme_names.index(phoneme_name))
                        label_phonemes.append(label_phoneme_names.index('-'))
                        #print(label_phonemes)
                        #exit(0)


                        tones=[]
                        toneTypes=[]
                        for i in range(sylla_num):
                            tone=[0,0,0]
                            label_tone=str(int(label_data['tone_'+str(i)][idx]))
                            for j in range(len(label_tone)):
                                if(label_tone[j]=='0'):
                                    tone[j]=6
                                else:
                                    tone[j]=int(label_tone[j])
                            #print(tone)
                            #exit(0)

                            # Tone type
                            toneType=None
                            if(6 in tone):
                                toneType=0
                            else:
                                if(tone[2]==0):
                                    if(tone[0]==tone[1]):
                                        toneType=1
                                    elif(tone[0]<tone[1]):
                                        toneType=2
                                    else:
                                        toneType=3
                                else:
                                    if(tone[0]==tone[1]):
                                        if(tone[2]>tone[1]):
                                            toneType=4
                                        else:
                                            toneType=5

                                    if(tone[1]==tone[2]):
                                        if(tone[0]<tone[1]):
                                            toneType=6
                                        else:
                                            toneType=7

                                    if(tone[0]>tone[1] and tone[1]<tone[2]):
                                        if(tone[2]>tone[0]):
                                            toneType=8
                                        elif(tone[2]==tone[0]):
                                            toneType=9
                                        else:
                                            toneType=10

                                    if(tone[0]<tone[1] and tone[1] > tone[2]):
                                        if (tone[2] > tone[0]):
                                            toneType = 11
                                        elif (tone[2] == tone[0]):
                                            toneType = 12
                                        else:
                                            toneType = 13

                            assert toneType != None, 'Wrong tone type!'

                            tones.extend(tone)
                            # print(tones)
                            # exit(0)

                            toneTypes.append(toneType)

                        features=scio.loadmat(filepath+'/features/'+file_name)['features']
                        #print(features.shape)
                        #exit(0)

                        speaker='_'.join(filepath.replace('\\','/').split('/')[-3:])

                        scio.savemat(save_name, {'features': features, 'word': word,
                                                 'sylla_num': sylla_num, 'label_phonemes': label_phonemes,
                                                 'tones':tones, 'toneType': toneTypes,
                                                 'speaker':speaker})

                        cnt+=1

                        #exit(0)

    print('%d file(s) has been processed.'%cnt)
