import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio

from models.Transformer import TransformerModel
from models.DenseLayers import DenseLayers

class Encoder(nn.Module):
    def __init__(self, target_len):
        super().__init__()

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.encoder = nn.GRU(input_size=64, hidden_size=128, num_layers=1, bias=True, batch_first=False, dropout=0,bidirectional=True)

        self.transformer_encoded = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, 256, dropout=0), 2)
        self.transformer_att_dense = nn.Linear(64, 256)
        self.transformer_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, 256, dropout=0), 4)

        self.sigmoid = nn.Sigmoid()

        self.target_len = target_len

    def forward(self, x):
        # Encoder
        encoded_seq, context = self.encoder(x)

        # Transformer for attention
        encoded_seq = self.transformer_encoded(encoded_seq)
        encoded_seq = encoded_seq[-self.target_len:, :, :]
        attention_seq = self.transformer_att(self.transformer_att_dense(x))
        attention_seq = attention_seq[-self.target_len:, :, :]
        attention_seq = self.sigmoid(attention_seq)

        return encoded_seq, attention_seq, context


class Decoder(nn.Module):
    def __init__(self, features_out, target_len):
        super().__init__()
        
        if(torch.cuda.is_available()):
            self.device=torch.device("cuda")
        else:
            self.device=torch.device("cpu")

        self.decoder_cell=nn.GRU(input_size=256, hidden_size=128, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)

        self.dense=DenseLayers(features_in=128, features_out=features_out)

        # Number of phonemes in each word
        self.target_len=target_len

    
    def forward(self, encoded_seq, attention_seq, context):
        batch_size = context.shape[1]
        ht = context[-1:, :, :]
        context = torch.mean(context, dim=0, keepdim=True)

        # Decoder
        #print(ht.shape, context.shape)
        #exit(0)
        pred=torch.empty(self.target_len, batch_size, 128)
        dec_in=torch.zeros_like(context)
        if (torch.cuda.is_available()):
            pred = pred.cuda()
            dec_in = dec_in.cuda()

        #print(dec_input.shape)
        if(torch.cuda.is_available()):
            pred=pred.cuda()
            dec_in=dec_in.cuda()

        for i in range(self.target_len):
            current_input=torch.cat((dec_in, context), dim=2)
            current_input=current_input+encoded_seq[min(i, encoded_seq.shape[0]-1),:,:]*attention_seq[min(i, attention_seq.shape[0]-1),:,:]
            output, ht=self.decoder_cell(current_input, ht)
            #print(output.shape, ht.shape)
            pred[i]=output.squeeze(0)
            dec_in=output
            #exit(0)

        #print(pred.shape)
        #exit(0)
        pred = self.dense(pred)
        #print(pred.shape)
        #exit(0)

        return pred


