# -*- coding: utf-8 -*-
"""
SequenceDataset: A custom PyTorch dataset class which consists of sequences of features stored in CSV files, 
                 with each file representing a different action.
                 
@author: Pei Yu Chou
"""

from torch.utils.data import Dataset
import os
import torch
import pandas as pd


class SequenceDataset(Dataset):
    def __init__(self, folder_path, sequence_length):
        self.folder_path = folder_path
        self.csv_files = []
        self.data = []
        self.labels = []
        self.classes = {}
        
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.csv'):
                self.csv_files.append(file_name)
                df = pd.read_csv(os.path.join(self.folder_path, file_name))
                for i in range(0, len(df), sequence_length):
                    sequence_data = df.iloc[i:i+sequence_length, :-1].values
                    sequence_label = df.iloc[i+sequence_length-1, -1]
                    if sequence_label not in self.classes:
                        self.classes[sequence_label] = len(self.classes)
                        
                    self.data.append(sequence_data)
                    self.labels.append(self.classes[sequence_label])
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, labels
    
    def print_dataset_info(self):
        print("Dataset Information:")
        print(f"\t CSV files loaded:{self.csv_files}")
        print(f"\t Classes:{self.classes}")
        print(f"\t Size of Dataset size:{len(self.data)}")
    
    def __str__(self):
        self.print_dataset_info()
        return ''
    