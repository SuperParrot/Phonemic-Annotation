import os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy.io as scio
import traceback


if __name__=='__main__':
    def get_phoneme_names(table_paths):
        phoneme_names = []

        for table_path in table_paths:
            f_read = open(table_path + '/phoneme_table.txt', encoding='utf-8')
            for line in f_read:
                phoneme_read = line.strip()
                if (not phoneme_read in phoneme_names):
                    phoneme_names.append(phoneme_read)
            f_read.close()

        return phoneme_names

    skip = False

    source_root_path = 'F:/Datasets/ChineseMarking/raw/train/'
    save_path = 'F:/Datasets/ChineseMarking/generated/train/'
    sample_rate=22050

    cnt=0

    label_phoneme_names = ['?','-']
    label_phoneme_names.extend(get_phoneme_names(['F:/Datasets/Zhongguoyuyan/']))
    print('Number of phonemes: %d' % len(label_phoneme_names))

    for part_name in os.listdir(source_root_path):
        part_path = source_root_path + '/' + part_name
        if (not os.path.isdir(part_path)):
            continue

        for data_name in os.listdir(part_path):
            data_path = part_path + '/' + data_name

            if (not os.path.isdir(data_path)):
                continue

            print(data_path)
            label_data = pd.read_excel(data_path + '/' + 'label.xlsx', sheet_name='Sheet1', header=0)
            #print(label_data['Word'])

            data_path = data_path + '/' + str(sample_rate) + '/features/'
            if(not os.path.isdir(data_path)):
                print('Found %s, but no features found.'%data_path)
                continue

            for file_name in os.listdir(data_path):
                if (not file_name.endswith('.mat')):
                    continue
                #print(file_name)

                if (not os.path.exists(save_path+part_name+'/'+data_name)):
                    os.makedirs(save_path+part_name+'/'+data_name)
                save_name=save_path+part_name+'/'+data_name+'/'+file_name.replace('.mat', '_featuresWithLabel.mat')
                #print(save_name)
                #exit(0)

                if (skip and os.path.exists(save_name)):
                    continue

                print('Processing %s...' % file_name)

                file_name_tmp=file_name
                for i in range(5):
                    file_name_tmp = file_name_tmp.replace('_timeAug' + str(i), '')
                    file_name_tmp = file_name_tmp.replace('_pitchAug' + str(i), '')
                idx=int(file_name_tmp.replace('.mat',''))-1

                word=label_data['Word'][idx]
                sylla_num=label_data['num_of_syllables'][idx]
                #print(word)

                label_phonemes = []
                for i in range(2 * sylla_num):
                    phonemes_name = str(label_data['phoneme_' + str(i)][idx]).strip()
                    if(phonemes_name=='-'):
                        continue
                    for phoneme_name in phonemes_name:
                        label_phonemes.append(label_phoneme_names.index(phoneme_name))
                label_phonemes.append(label_phoneme_names.index('-'))
                #print(label_phonemes)
                #exit(0)

                tones = []
                toneTypes = []
                for i in range(sylla_num):
                    tone = [0, 0, 0]
                    label_tone = str(int(label_data['tone_' + str(i)][idx]))
                    for j in range(len(label_tone)):
                        if (label_tone[j] == '0'):
                            tone[j] = 6
                        else:
                            tone[j] = int(label_tone[j])
                    # print(tone)
                    # exit(0)

                    # Tone type
                    toneType = None
                    if (6 in tone):
                        toneType = 0
                    else:
                        if (tone[2] == 0):
                            if (tone[0] == tone[1]):
                                toneType = 1
                            elif (tone[0] < tone[1]):
                                toneType = 2
                            else:
                                toneType = 3
                        else:
                            if (tone[0] == tone[1]):
                                if (tone[2] > tone[1]):
                                    toneType = 4
                                else:
                                    toneType = 5

                            if (tone[1] == tone[2]):
                                if (tone[0] < tone[1]):
                                    toneType = 6
                                else:
                                    toneType = 7

                            if (tone[0] > tone[1] and tone[1] < tone[2]):
                                if (tone[2] > tone[0]):
                                    toneType = 8
                                elif (tone[2] == tone[0]):
                                    toneType = 9
                                else:
                                    toneType = 10

                            if (tone[0] < tone[1] and tone[1] > tone[2]):
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

                features=scio.loadmat(data_path+'/'+file_name)['features']
                #print(features.shape)

                #print(part_path)
                speaker = '_'.join(part_path.replace('\\', '/').replace('//','/').split('/')[-2:])+'_'+data_name
                #print(speaker)
                #exit(0)

                #print(save_name)
                #exit(0)
                if(os.path.exists(save_name)):
                    os.remove(save_name)
                scio.savemat(save_name, {'features': features, 'word': word,
                                         'sylla_num': sylla_num, 'label_phonemes': label_phonemes,
                                         'tones': tones, 'toneType': toneTypes,
                                         'speaker': speaker})

                cnt+=1

                #exit(0)

    print('%d file(s) has been processed.'%cnt)
