import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as scio

    

class DenseLayers(nn.Module):
    def __init__(self, features_in, features_out):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(features_in, features_in),
            nn.LayerNorm(features_in),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(features_in, features_in//2),
            nn.LayerNorm(features_in//2),
            nn.GELU(),
            
            nn.Linear(features_in//2, 64),
            nn.GELU(),
        )

        self.fc_end = nn.Linear(64, features_out)

    def forward(self, x):
        x = self.fc(x)
        output = self.fc_end(x)
        
        return output

    

