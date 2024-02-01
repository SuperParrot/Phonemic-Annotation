import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.CTCLossDecoder import prefix_beam_decode, remove_blank
import matplotlib.pyplot as plt
import time
import os
import random
from tqdm import tqdm
import numpy as np
import itertools

from dataset import MyDataset

from models.Model import Net

from models.utils.LossFuncs import *

from utils.Metrics import *
from utils.CTCLossDecoder import *

class collater():
    def __init__(self, *params):
        self.params = params

    def __call__(self, tuple_in):
        x = [item[0] for item in tuple_in]
        lengths = [len(item[0]) for item in tuple_in]
        label_phonemes = list(itertools.chain.from_iterable([item[1] for item in tuple_in]))
        label_toneType = list(itertools.chain.from_iterable([item[2] for item in tuple_in]))

        label_phonemes_lengths = [len(item[1]) for item in tuple_in]
        label_toneType_lengths = [len(item[2]) for item in tuple_in]

        word = [item[3] for item in tuple_in]

        del tuple_in

        #print(x[0].shape)
        x = nn.utils.rnn.pad_sequence(x, batch_first=False)

        assert sum(label_phonemes_lengths)==len(label_phonemes), 'Problems in lengths of phoneme label.'
        assert sum(label_toneType_lengths)==len(label_toneType), 'Problems in lengths of tone type label.'

        label_phonemes = torch.as_tensor(label_phonemes, dtype=torch.long)

        label_toneType = torch.as_tensor(label_toneType, dtype=torch.long)

        #print(x.shape,label_phonemes.shape)
        #print(lengths)
        #exit(0)

        #print(x.shape)

        return x, label_phonemes, label_toneType, label_phonemes_lengths, label_toneType_lengths, word

class Interface():
    def __init__(self):
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")
            
        self.model_path='./params_saved'
        '''
        seed = 3
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)
        '''

    def generateModel(self):
        print('Creating new model...')
        net=Net()
        if(torch.cuda.is_available()):
            net=net.to(self.device)
        
        if(not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)
        torch.save(net.state_dict(), self.model_path+'/params.pkl')
        print('New model saved.')

    def train(self, epoch=1, learning_rate=1e-5, batch_size = 1, save_freq=1, shuffle=True, train_mode=''):
        self.net=Net()
        if(torch.cuda.is_available()):
            try:
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl'))
            except:
                self.generateModel()
        else:
            try:
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl', map_location='cpu'))
            except:
                self.generateModel()

        self.net.to(self.device)
        self.net.train()
        
        self.ctcLoss=nn.CTCLoss(blank=0)
        
        '''
        for name, para in other_params:
            print('-->name:', name)
        exit(0)
        '''
        
        optim_list=[]
        optim_list.append({'params': self.net.parameters(), 'lr': learning_rate, 'weight_decay': 1e-4})
        
        self.optimizer=optim.Adam(optim_list)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epoch, eta_min=1e-6, last_epoch=-1)
        
        #print(torch.cuda.device_count())
        device_ids = range(torch.cuda.device_count())
        if(len(device_ids)>1):
            self.net = torch.nn.DataParallel(self.net, device_ids=[0,1,2])
        else:
            torch.cuda.set_device(0)
        
        self.train_data=MyDataset(file_name='train_list.txt')
        batchNum=self.train_data.__len__()//batch_size
        if(self.train_data.__len__() % batch_size!=0):
            batchNum+=1

        progressPerPrint=1/(epoch*batchNum)
        progress=0

        acc_last=0
        acc_decreaseCnt=0
        last_time=-1

        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=shuffle, collate_fn = collater(None), num_workers=2, drop_last=False, pin_memory=True)
        for i in range(epoch):
            if(torch.cuda.is_available()):
                torch.cuda.empty_cache()

            epoch_aveLoss=0
            current_loss=0
            current_loss_batch_cnt=0

            if (train_mode == 'Alternation'):
                work_mode='PBranch_Enc'
            else:
                work_mode=train_mode
            for batch_idx, (x, label_phonemes, label_toneType, label_phonemes_lengths, label_toneType_lengths, word)\
                    in enumerate(train_loader):
                if(train_mode=='Alternation'):
                    if(batch_idx%10==0):
                        if(work_mode=='PBranch_Enc'):
                            work_mode='PBranch_Disc'
                        elif(work_mode=='PBranch_Disc'):
                            work_mode='PBranch_Enc'

                loss_P, loss_T=self.train_batch(x, label_phonemes, label_toneType, label_phonemes_lengths, label_toneType_lengths, work_mode)
                current_loss+=loss_P+loss_T
                current_loss_batch_cnt+=1

                if(last_time>0):
                    speed=progressPerPrint/(time.time()-last_time)
                    eta=(1.0-progress)/speed
                    m, s = divmod(eta, 60)
                    h, m = divmod(m, 60)
                        
                    if((batch_idx % 10 == 0 and batch_idx > 0) or batch_idx >= batchNum - 1):
                        print("epoch:%d/%d batch:%d/%d\tloss P:%.4lf loss T:%.4lf ETA:%02dH:%02dM:%02dS"%(i+1, epoch, batch_idx+1, batchNum, loss_P, loss_T, h,m,s))
                else:
                    print("epoch:%d/%d batch:%d/%d\tloss P:%.4lf loss T:%.4lf ETA:??H:??M:??S"%(i+1, epoch, batch_idx+1, batchNum, loss_P, loss_T))
                progress+=progressPerPrint
                last_time=time.time()
                
                if((batch_idx % 10 == 0 and batch_idx > 0) or batch_idx >= batchNum - 1):
                    loss=current_loss.cpu().detach().numpy()
                    epoch_aveLoss+=loss
                    
                    current_loss_batch_cnt=0
                    current_loss = 0

            print("epoch %d has finished. Average loss is: %.6lf"%(i+1, epoch_aveLoss/batchNum))
            
            self.scheduler.step()
            
            if(i%save_freq==0 or i==epoch-1):
                print('Saving...')
                try:
                    torch.save(self.net.module.state_dict(), self.model_path+'/params.pkl')
                    torch.save(self.net.module.state_dict(), self.model_path+'/params_'+str(i)+'.pkl')
                except:
                    torch.save(self.net.state_dict(), self.model_path+'/params.pkl')
                    torch.save(self.net.state_dict(), self.model_path+'/params_'+str(i)+'.pkl')
                print('Saved.')
                

    def train_batch(self, x, label_phonemes, label_toneType, label_phonemes_lengths, label_toneType_lengths, work_mode):
        if(torch.cuda.is_available()):
            x=x.cuda()
            label_phonemes=label_phonemes.cuda()
            label_toneType=label_toneType.cuda()

        pred_P, pred_T=self.net(x, work_mode)
        log_pred_P=pred_P.log_softmax(-1)
        log_pred_T=pred_T.log_softmax(-1)

        #print(log_pred.shape, label.shape)
        #exit(0)
        input_lengths=[]
        for i in range(log_pred_P.shape[1]):
            input_lengths.append(log_pred_P.shape[0])
        #print(input_lengths, label_lengths)
        #print(len(label), sum(label_lengths))
        #exit(0)
        loss_P = self.ctcLoss(log_probs=log_pred_P, targets=label_phonemes, input_lengths=input_lengths, target_lengths=label_phonemes_lengths)

        input_lengths = []
        for i in range(log_pred_T.shape[1]):
            input_lengths.append(log_pred_T.shape[0])
        loss_T = self.ctcLoss(log_probs=log_pred_T, targets=label_toneType, input_lengths=input_lengths,target_lengths=label_toneType_lengths)

        #print(loss_P, loss_T)
        #exit(0)
        if(work_mode == 'PBranch_Enc'):
            loss = loss_P - loss_T
        elif(work_mode == 'PBranch_Disc'):
            loss = loss_P + loss_T

        self.optimizer.zero_grad()
        loss.backward()
        '''
        for name, parms in self.net.named_parameters():
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad.max())
        exit(0)
        '''
        self.optimizer.step()
        #exit(0)
        
        return loss_P, loss_T

    def eval(self, eval_loader):
        self.net.eval()
        
        F_levenDis=LevenshteinDis()
        accuracy=0
        pred_syllas_all, label_syllas_all = [], []
        
        tot_cnt=0
        corrrect_cnt=0
        
        for batch_idx, (x, label_phonemes, label_toneType, label_phonemes_lengths, label_toneType_lengths, word) in enumerate(eval_loader):
            if(torch.cuda.is_available()):
                x=x.cuda()
                label_phonemes=label_phonemes.cuda()
                label_toneType=label_toneType.cuda()

            with torch.no_grad():
                pred_phonemes, pred_tones=self.net(x, 'All')

            #print(pred_phonemes.shape)
            #exit(0)

            #Greedy decoding
            pred_phonemes_decoded=torch.argmax(pred_phonemes,dim=-1)
            pred_phonemes_decoded=pred_phonemes_decoded.permute(1,0)


            #Prefix beam decoding
            '''
            pred_phonemes=F.softmax(pred_phonemes, dim=-1)

            pred_phonemes_decoded=[]
            for i in range(0, pred_phonemes.shape[1]):
                pred_phonemes_batch=pred_phonemes[:,i,:].cpu()
                beam_test = prefix_beam_decode(pred_phonemes_batch, beam_size=3)
                # for beam_string, beam_score in beam_test:
                # print(remove_blank(beam_string), beam_score)
                #exit(0)

                pred_phonemes_decoded.append(beam_test[0][0])
            '''

            #print(pred_phonemes_decoded[0])
            #exit(0)

            pred_syllas, label_syllas=[], []

            label_len_cnt=0
            for i in range(len(pred_phonemes_decoded)):
                pred_phonemes_vis, label_phonemes_vis = '', ''
                cur_ch='-'
                for j in range(len(pred_phonemes_decoded[i])):
                    #print(pred_phonemes[i])
                    pred_ch=self.label_phoneme_names[pred_phonemes_decoded[i][j]]
                    if(pred_ch=='-' or pred_ch=='?'):
                        continue
                    if(pred_ch!=cur_ch):
                        pred_phonemes_vis+=pred_ch
                    cur_ch=pred_ch

                for j in range(label_phonemes_lengths[i]):
                    label_phonemes_vis += self.label_phoneme_names[label_phonemes[label_len_cnt+j]].replace('-','')
                label_len_cnt+=label_phonemes_lengths[i]

                #print(word[i], pred_phonemes_vis, label_phonemes_vis)
                if(pred_phonemes_vis!=label_phonemes_vis):
                    #print(word[i], pred_phonemes_vis, label_phonemes_vis)
                    pass
                else:
                    corrrect_cnt+=1
                #exit(0)
                pred_syllas.append(pred_phonemes_vis)
                label_syllas.append(label_phonemes_vis.replace(' ', ''))

                levenDis=F_levenDis(pred_phonemes_vis, label_phonemes_vis)
                accuracy+=max(1-levenDis/len(label_phonemes_vis), 0)
                #print(max(1-levenDis/len(label_phonemes_vis), 0))

                #print(levenDis, 1-levenDis/len(label_phonemes_vis))
                #print(accuracy)
            #exit(0)

            #print(pred_syllas, label_syllas)
            #exit(0)
            
            tot_cnt+=len(label_syllas)

            #exit(0)
            #print(tot_correct_cnt, tot_cnt_table)
            '''
            if(batch_idx>2):
                break
            '''

        print(corrrect_cnt/tot_cnt)
        accuracy/=tot_cnt
        
        return accuracy

    def predict(self, file_name, assigned_modelName=None):
        
        seed = 3
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)
        
        self.net=Net()
        '''
        for name, parms in self.net.named_parameters():
            print(name, '-->grad_requirs:',parms.requires_grad)
            #print('-->name:', name)
        exit(0)
        '''
        test_data=MyDataset(file_name=file_name)
        test_loader = DataLoader(dataset=test_data, batch_size=20, shuffle=False, collate_fn=collater(None))

        self.label_phoneme_names=test_data.label_phoneme_names
    
        if(assigned_modelName==None):
            if(torch.cuda.is_available()):
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl'))
                torch.cuda.set_device(0)
                self.net=self.net.to(self.device)
            else:
                self.net.load_state_dict(torch.load(self.model_path+'/params.pkl', map_location='cpu'))
        else:
            if (torch.cuda.is_available()):
                self.net.load_state_dict(torch.load(self.model_path + '/' + assigned_modelName))
                torch.cuda.set_device(0)
                self.net = self.net.to(self.device)
            else:
                self.net.load_state_dict(torch.load(self.model_path + '/' + assigned_modelName, map_location='cpu'))
    
        return self.eval(test_loader)
        