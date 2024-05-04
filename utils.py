# -*- coding: utf-8 -*-
"""
@author: Pei Yu Chou
"""

from matplotlib import pyplot as plt
import torch


def print_args(args):
    print("Arguments entered:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
def print_error_in_red(msg):
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print(f'{FAIL}{msg}{ENDC}')  # ANSI escape code for red text
    
    
class Logger():
    def __init__(self, result_dir, run_name, monitor_metric='loss', verbose=1):
        self.result_dir = result_dir
        self.run_name = run_name
        self.monitor_metric = monitor_metric
        if self.monitor_metric == 'loss':
            self.best_monitor_metric = float('inf')
        elif self.monitor_metric == 'accuracy':
            self.best_monitor_metric = 0.0
        else:
            raise ValueError('Invalid metric, choose either loss or accuracy')
        self.metric = {'train_loss': [],'val_loss': [],'train_acc': [],'val_acc': []}
        self.model_checkpoint = {}
        self.best_epoch = 0
        self.verbose = verbose
    
    def print_metrics(self):
        for key, value in self.metric.items():
            print(f'\t{key}:{value[-1]:.4f}', end='')
        print('\n')
                    
    def update_metrics(self, train_loss, val_loss, train_acc, val_acc):
        self.metric['train_loss'].append(train_loss)
        self.metric['val_loss'].append(val_loss)
        self.metric['train_acc'].append(train_acc)
        self.metric['val_acc'].append(val_acc)
        if self.verbose:
            self.print_metrics()

    def check_update_improve_loss(self):
        if self.best_monitor_metric > self.metric['val_loss'][-1]:
            self.best_monitor_metric = self.metric['val_loss'][-1]
            return True
        else:
            return False
        
    def check_update_improve_acc(self):
        if self.best_monitor_metric < self.metric['val_acc'][-1]:
            self.best_monitor_metric = self.metric['val_acc'][-1]
            return True
        else:
            return False

    def update_monitor_metric(self):
        if self.monitor_metric == 'loss':
            return self.check_update_improve_loss()
        if self.monitor_metric == 'accuracy':
            return self.check_update_improve_acc()
        
    def save_checkpoint(self, checkpoint, best_epoch):
        self.model_checkpoint = checkpoint
        self.best_epoch = best_epoch
        torch.save(self.model_checkpoint, f'{self.result_dir}/{self.run_name}_checkpoint.pt')
    
    def print_training_result(self):
        print('Model Result:')
        for key, value in self.metric.items():
            print(f'\t{key}:{value[self.best_epoch]:.4f}')
        print('\n')
        
        
    def save(self):
        torch.save(self, f'{self.result_dir}/{self.run_name}_logger.pt')

    def plot_loss_curve(self, save=True):
        epochs = len(self.metric['train_loss'])
        plt.plot(range(1, epochs+1), self.metric['train_loss'], label="train")
        plt.plot(range(1, epochs+1), self.metric['val_loss'], label="validation")
        plt.title("Loss curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if save:
            plt.savefig(f'{self.result_dir}/{self.run_name}_loss_curve.png')
        plt.show() # save it before calling show(), cuz show() clears the plot.
    
    def plot_acc_curve(self, save=True):
        epochs = len(self.metric['train_acc'])
        plt.plot(range(1, epochs+1), self.metric['train_acc'], label="train")
        plt.plot(range(1, epochs+1), self.metric['val_acc'], label="validation")
        plt.title("Accuracy curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1.05])
        plt.legend()
        if save:
            plt.savefig(f'{self.result_dir}/{self.run_name}_acc_curve.png')
        plt.show()