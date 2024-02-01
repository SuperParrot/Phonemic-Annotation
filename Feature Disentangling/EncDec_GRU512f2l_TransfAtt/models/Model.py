import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio
from thop import profile
from thop import clever_format

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

        '''
        macs_tot, params_tot = 0, 0
        x = torch.randn(1, 1, 64)

        macs, params = profile(self.P_encoder, inputs=(x,))
        macs_tot += macs
        params_tot += params

        macs, params = profile(self.T_encoder, inputs=(x,))
        macs_tot += macs
        params_tot += params

        with torch.no_grad():
            encoded_seq_P, attention_seq_P, context_P = self.P_encoder(x)
            encoded_seq_T, attention_seq_T, context_T = self.T_encoder(x)

        if(torch.cuda.is_available()):
            encoded_seq_P=encoded_seq_P.cuda()
            attention_seq_P=attention_seq_P.cuda()
            context_P=context_P.cuda()

            encoded_seq_T=encoded_seq_T.cuda()
            attention_seq_T = attention_seq_T.cuda()
            context_T = context_T.cuda()

            self.P_decoder=self.P_decoder.cuda()
            self.T_decoder=self.T_decoder.cuda()


        macs, params = profile(self.P_decoder, inputs=(encoded_seq_P, attention_seq_P, context_P))
        macs_tot += macs
        params_tot += params

        macs, params = profile(self.T_decoder, inputs=(encoded_seq_T, attention_seq_T, context_T))
        macs_tot += macs
        params_tot += params

        macs, params = clever_format([macs_tot, params_tot], "%.3f")
        # macs: 281.256M params: 28.196M
        print(macs, params)
        exit(0)
        '''

    
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


