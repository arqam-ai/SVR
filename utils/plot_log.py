#!/usr/bin/python
'''
plot_log in DeepGeom

author  : cfeng
created : 1/31/18 7:49 AM
'''

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

#from utils import TrainTestMonitor as TTMon
#from Tkinter import Tk
#import tkFileDialog

def main(args):

#    address_0 = 'baseline_lossplot'
    address_0 = 'ae_label_baseline'
    
    results_list = ['stats_finecd_epochval.npz', 'stats_finecd_itertrain.npz']
    results_list = [results_list[0]]

#    logs_list = ['logs_aebatch10_0', 'logs_aebatch10_1', 'logs_aebatch10_2', 'logs_aebatch128_0'] #'logs',logs_adam_0 logs_adam_1
    logs_list = ['logs', 'logs_adam_0', 'logs_adam_1']
    logs_list = logs_list[3:4]

    fig = plt.figure(figsize=(10,10))

    for row, log in enumerate(logs_list):   
        for col, result in enumerate(results_list): 
            file = os.path.join(address_0, '{}/{}'.format(log,result))
            stats = np.load(file)
            plt.plot(stats["iter_loss"][:,0], stats["iter_loss"][:,1],'-', label = '{}'.format('Auto-encoder'))
            plt.tick_params(labelsize=33)
            plt.legend(fontsize=20)
    plt.show()

#   plt.savefig(os.path.join(args.savedir,'ae_label_adam_No2.png'))
#   plt.savefig(os.path.join(args.savedir,'label5_ae_0.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('-d','--log_dir',type=str, default='../experiment/mixup_alpha1/logs', help='log folder')
    parser.add_argument('-e','--plot_extra',type=int, default=0, help='plot training accuracy and test loss')
    parser.add_argument('-s','--savedir',type=str, default="../img/mixup", help='path to save plot')
    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
