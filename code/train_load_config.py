'''
Author: PikachuLitt 1951689929@qq.com
Date: 2023-04-15 09:46:54
LastEditors: litt 1951689929@qq.com
LastEditTime: 2024-06-12 16:24:46
FilePath: /quantum_code/quantum_encoding/models/model.py
Description: 有噪声的模型训练
'''
import os
import sys
import yaml
import random
import torch
import argparse
import subprocess
import datetime
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.entire_circ import EmptyEncoder, TempleteCirc
from models.classify_circ import classify_circ_dict
from utils.util import set_seed

if __name__ == '__main__':
    set_seed(42)

    arch = {'n_wires': 4, \
            'static': False,
            'epochs': 30,
            'enc_layer': 'general', 'use_qiskit': False, \
            'train_model_dir': '../quantum_encoding_data/train_models/test_del',
            'valid_model_epochs': True,
            'save_epoch': False,
            "rotation_gate": "rz", # toward IQP encoding
            'enc_blocks':1, 'cls_blocks':2, # 'cls_blocks'
            'encoded_states_dir_name': 'encoded_states_test',} 

    now_date = datetime.datetime.now().strftime('%m%d') 
    
    for _, (enc_block, cls_block) in enumerate([[1, 1],[1, 2], [2, 2], [3, 2]]):
        print(enc_block, cls_block)
        arch['enc_blocks'], arch['cls_blocks'] = enc_block, cls_block
        arch['train_model_dir'] = os.path.join(arch['train_model_dir'], \
                            'enc_b'+str(arch['enc_blocks']) + '_cls_b' + str(arch['cls_blocks']))
        arch['train_res_file'] = os.path.join(arch['train_model_dir'], \
                                        '0_train_result', 'train_'+ str(now_date)+'.txt')
        if not os.path.exists(os.path.dirname(arch['train_res_file'])):
            os.makedirs(os.path.dirname(arch['train_res_file']), exist_ok=True)
        yml_path = os.path.join(arch['train_model_dir'], 'configs')
        if not os.path.exists(yml_path):
            os.makedirs(yml_path, exist_ok=True)
        with open(arch['train_res_file'], 'a+') as file:
            file.write('enc_layer\tdataset\t' + 'model_type\tepoch\tmodel\t' + \
                        'valid_acc\tvalid_loss\t' + \
                        'test_acc\t' + 'test_loss\n')

        ################################# Initialize train configuration #################################
        datasets = ["mnist", "fashion", "cifar10"]
        enc_layers = ['amplitude', 'state', 'phase', 'general', 'multiphase', 'multiphase_z', 'IQP', 'IQP_X', "IQP_Y"]
        model_list = list(classify_circ_dict.keys())

        # ”state“ and "amplitude" encoding just apply 2**num_qubits classical bits
        if "state_new" in enc_layers and arch['enc_blocks'] > 1:
            enc_layers.remove("state_new") # state just encodes 2**n_wires bits
        if "amplitude" in enc_layers and arch['enc_blocks'] > 1:
            enc_layers.remove("amplitude")

        ################################# train begining #################################
        for dataset in datasets:
            arch['dataset'] = dataset
            
            for cls_circ in model_list: 
                for enc_layer in enc_layers:
                    # for model_type in ['templete', 'empty']:
                    for model_type in ['templete']:
                        arch['classify_circ'] = cls_circ
                        arch['enc_layer'] = enc_layer
                        
                        if dataset == "vowel":
                            arch['data_type'] = "vowel"
                        elif dataset in ['imdb', "reuters"]:
                            arch['data_type'] = "nlp"
                        else:
                            arch['data_type'] = "image"

                        arch['encode_method'] = "4x4_ryzxy" if enc_layer == "general" else ''
                        arch['n_wires'] = 16 if arch['encode_method'] == "16x1_ry" else 4
                        arch['filepath'] = os.path.join(arch['train_model_dir'],
                                                        str(arch['dataset']) + "_" + enc_layer + '_' + \
                                                        arch['classify_circ'] + "/")

                        if model_type == 'templete':
                            ########## train templete ########## 
                            print(f"\n\nrun templete: {dataset} {enc_layer} {arch['encode_method']}")
                            arch['train_empty'] = False
                            yml_file = os.path.join(yml_path, str(arch['dataset']) + "_" + \
                                                    enc_layer + '_' + arch['classify_circ']+'.yml')
                        elif model_type == 'empty':
                            ########## train empty ##########
                            print(f"\n\nrun empty: {dataset} {enc_layer} {arch['encode_method']}")
                            arch['train_empty'] = True
                            arch['filepath'] = os.path.join(arch['train_model_dir'],
                                                            str(arch['dataset']) + "_empty_" + enc_layer + '_' + \
                                                            arch['classify_circ'] + "/")
                            yml_file = os.path.join(yml_path, str(arch['dataset']) + "_empty_" + \
                                                    enc_layer + '_' + arch['classify_circ']+'.yml')
                        
                        with open(yml_file, 'w') as f:
                            yaml.dump(arch, f)
                        
                        exe_res = os.system(f"python train.py {yml_file}")
                        if exe_res != 0:
                            print("="*50+" error "+"="*50)
                            sys.exit(0)