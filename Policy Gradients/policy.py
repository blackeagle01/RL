import torch
import torch.nn as nn

from torch.nn import functional as F

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense1= nn.Linear(4,15)
        self.dense2= nn.Linear(15,15)
        
        self.dense3= nn.Linear(15,2)
        
        
    def forward(self,x):
        
        out = self.dense1(x)
        out = F.relu(out)
        
        out = F.dropout(out)
        
        out = self.dense2(out)
        out = F.relu(out)
        
        out = self.dense3(out)
        out = F.softmax(out,dim=-1)

       	return out
        
