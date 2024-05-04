# -*- coding: utf-8 -*-
"""
A simple LSTM model implemented using PyTorch

@author: Pei Yu Chou
"""
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out:(batch_size, seq_length, hidden_size)
        
        # take last timestamp of LSTM
        out = self.fc(out[:, -1, :])  # out:(batch_size, num_classes)
        return out
