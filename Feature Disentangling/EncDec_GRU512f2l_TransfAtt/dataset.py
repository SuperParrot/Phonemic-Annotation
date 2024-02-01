import numpy as np
import torch
import scipy.io as scio
import h5py
from PIL import Image
from PIL import ImageChops
import matplotlib.pyplot as plt
import random
import os
import tarfile, gzip
import traceback
import librosa as librosa

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, data_num=-1):
        super(MyDataset,self).__init__()

        self.label_phoneme_names=['?','-']
        #self.label_phoneme_names.extend(self.get_phoneme_names(['/data4T/JBW/']))
        self.label_phoneme_names.extend(self.get_phoneme_names(['F:/Datasets/Zhongguoyuyan/']))
        #print(self.label_phoneme_names)
        print('Number of phonemes: %d'%len(self.label_phoneme_names))

        self.data_names=self.load_dataList(file_name, data_num)
        print('%d data has been succesfully loaded.'%len(self.data_names))

        self.sample_rate=22050

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def un_gz(self, file_name, save_name):
        f_name = file_name.replace(".gz", "")
        g_file = gzip.GzipFile(file_name)
        open(save_name, "wb+").write(g_file.read())
        g_file.close()

    def untar(self, fname, dirs):
        t = tarfile.open(fname)
        t.extractall(path = dirs)

    def load_dataList(self, file_name, data_num):
        data_names=[]
        
        f=open('./data_list/'+file_name, 'r', encoding = 'utf-8')
        for data_name in f.readlines():
            data_name=data_name.rstrip('\n').replace('\\','/')

            '''
            if(not ('Zhonglou' in data_name)):
                continue
            '''
            data_names.append(data_name)
            if(data_num>0 and len(data_names)>=data_num):
                break
        f.close()

        return data_names

    def get_phoneme_names(self, table_paths):
        phoneme_names=[]

        for table_path in table_paths:
            f_read = open(table_path + '/phoneme_table.txt', encoding='utf-8')
            for line in f_read:
                phoneme_read = line.strip()
                if (not phoneme_read in phoneme_names):
                    phoneme_names.append(phoneme_read)
            f_read.close()

        return phoneme_names

    def __getitem__(self, index):
        x, label_phonemes, label_toneType, word=self.get_data(index)
        
        return (x, label_phonemes, label_toneType, word)


    def get_data(self, index):
        data_name = self.data_names[index % len(self.data_names)]
        input_mat = scio.loadmat(data_name)
        # input_mat=h5py.File(data_name)

        # ['features', 'word', 'sylla_num', 'syllas']
        # print(input_mat.keys())
        # exit(0)

        x = torch.Tensor(input_mat['features']).permute(1, 0)
        label_phonemes = input_mat['label_phonemes'][0]
        label_toneType = input_mat['toneType'][0] + 1 # Spare for blank labels!
        word = input_mat['word'][0]

        del input_mat

        # print(x.shape)
        # print(label_phonemes_name)
        # exit(0)

        # print(word)
        # print(label_phonemes_name)
        # print(label_phonemes)
        # exit(0)

        label_phonemes = torch.tensor(label_phonemes).long()

        label_toneType = torch.tensor(label_toneType).long()

        return x, label_phonemes, label_toneType, word

    def __len__(self):
        return len(self.data_names)

class GetDataList():
    def __init__(self):
        pass
        
    def get_data(self, save_path, save_name, data_paths):
        data_names=[]
        for data_path in data_paths:
            data_names.extend(self.read_dataList(data_path))
            print(self.read_dataList(data_path))
            exit(0)
        
        tot=len(data_names)
        print('%d data has been succesfully detected.'%tot)

        self.writeToTxt(save_path, save_name, data_names)
        
        
    def writeToTxt(self, save_path, file_name, data_list):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path)
    
        if os.path.exists(save_path+'/'+file_name):
            print('Warning: %s has already existed.'%file_name)
            os.remove(save_path+'/'+file_name)
        f = open(save_path+'/'+file_name,'w')
        for data_name in data_list:
            f.write(data_name+'\n')
        f.close()

    def read_dataList(self, path):
        all_files=[]

        for filepath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if(not filename.endswith('_featuresWithLabel.mat')):
                    continue

                #print(os.path.join(filepath, filename))
                all_files.append(os.path.join(filepath, filename).replace('\\', '/'))

        return all_files

    def un_gz(self, file_name, save_name):
        f_name = file_name.replace(".gz", "")
        g_file = gzip.GzipFile(file_name)
        open(save_name, "wb+").write(g_file.read())
        g_file.close()

    def untar(self, fname, dirs):
        t = tarfile.open(fname)
        t.extractall(path = dirs)

    
