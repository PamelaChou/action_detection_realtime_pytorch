# -*- coding: utf-8 -*-
"""
Sign Detection using LSTM Model main.py

This script trains an LSTM model for action detection.
It utilizes sequences of features extracted from body actions to predict the corresponding action or sign.

@author: Pei Yu Chou
"""

import os
import argparse
from dataset import SequenceDataset
from utils import Logger, print_args, print_error_in_red
from model import LSTMModel
from train import train_model
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def get_args_parser():
    parser = argparse.ArgumentParser('Train LSTM model for action detection', add_help=False)
    parser.add_argument('data_dir', default='data/',type=str,
                        help='Path of training data with csv files.')
    parser.add_argument('run_name', type=str,
                        help='Name of this run')
    parser.add_argument('--result_dir', default='result', type=str,
                        help='Path of results')
    parser.add_argument('--seqence_length', default=20, type=int,
                        help='Length of seqence for each action')
    parser.add_argument('--train_batch_size', default=20, type=int) 
    parser.add_argument('--val_batch_size', default=10, type=int) 
    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Print details')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save result plot in local')
    
    return parser


def main(args):
    print_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device : {device}')
    os.makedirs(os.path.join(os.getcwd(), args.result_dir), exist_ok=True)
    
    ## data
    dataset = SequenceDataset(args.data_dir, args.seqence_length)
    print(dataset)
    classes:dict = dataset.classes
    train_data, val_data = train_test_split(dataset, test_size=0.2, shuffle=True, 
                                            stratify=dataset.labels)
    
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False)
    input_size = next(iter(val_dataloader))[0].shape[2]
    
    ## Model
    model_param = {"input_size": input_size,
                   "hidden_size": 64,
                   "num_layers": 2,
                   "num_classes": len(classes),
                   "classes": classes,
                   "seqence_length": args.seqence_length}
    
    model = LSTMModel(input_size=input_size, 
                      hidden_size=model_param['hidden_size'], 
                      num_layers=model_param['num_layers'], 
                      num_classes=model_param['num_classes'])
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    logger = Logger(args.result_dir, args.run_name, verbose=args.verbose)
    train_model(model, model_param, args.epochs, device, train_dataloader, val_dataloader,
                criterion, optimizer, logger, args.run_name)
    
    logger.plot_loss_curve(save=args.save_plot)
    logger.plot_acc_curve(save=args.save_plot)
    
    
if __name__ == '__main__':
    try:
        parser = get_args_parser()
        args = parser.parse_args()        
        main(args)
    except Exception as error:
        print_error_in_red(f'{type(error).__name__} - {error}')
        raise error

