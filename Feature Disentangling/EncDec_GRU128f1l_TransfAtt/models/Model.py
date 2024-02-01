import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio

from models.EncDec import Encoder, Decoder

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")

        target_len_P=(6*4)*2+1
        target_len_T=5*2+1

        self.P_encoder=Encoder(target_len=target_len_P)
        self.P_decoder=Decoder(features_out=66, target_len=target_len_P)

        self.T_encoder = Encoder(target_len=target_len_T)
        self.T_decoder = Decoder(features_out=15, target_len=target_len_T)

    
    def forward(self, x, work_mode=None):
        if(work_mode=='All' or work_mode==None):
            encoded_seq_P, attention_seq_P, context_P=self.P_encoder(x)
            encoded_seq_T, attention_seq_T, context_T = self.T_encoder(x)

            pred_P=self.P_decoder(encoded_seq_P, attention_seq_P, context_P)
            pred_T=self.T_decoder(encoded_seq_T, attention_seq_T, context_T)

        elif(work_mode=='PBranch_Enc'):
            encoded_seq_P, attention_seq_P, context_P = self.P_encoder(x)

            pred_P=self.P_decoder(encoded_seq_P, attention_seq_P, context_P)

            with torch.no_grad():
                pred_T=self.T_decoder(encoded_seq_P, attention_seq_P, context_P)

        elif(work_mode=='PBranch_Disc'):
            with torch.no_grad():
                encoded_seq_P, attention_seq_P, context_P = self.P_encoder(x)

            pred_P=self.P_decoder(encoded_seq_P, attention_seq_P, context_P)
            pred_T=self.T_decoder(encoded_seq_P, attention_seq_P, context_P)


        #print(pred_P.shape, pred_T.shape)
        #exit(0)

        return pred_P, pred_T


