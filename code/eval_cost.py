import os
import random
import torch
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from qiskit import QuantumCircuit

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugins import (tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,)
from models.classify_circ import classify_circ_dict
from models.enc_circ import encoder_circ_dict, encoder_op_list_name_dict
from utils.util import PCA_reduction
from qiskit_core.qiskit_processor import qiskit_assemble_circs_new, op_history2qiskit_expand_params_new

from models.entire_circ import TempleteCirc
from models.classify_circ import classify_circ_dict
from utils.util import set_seed, get_dataset, valid_model

def load_model(arch):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if arch['valid_model_epochs']:
        for epoch in range(1, 51, 1):
            arch['epoch'] = epoch
            # arch['model_file'] = model_path + pth_file
            arch['model_file'] = os.path.join(arch['model_path'], arch['dataset'] + '_' + \
                arch['encode_method'] + '_' + arch['classify_circ'] + '_epoch' + str(epoch) + '.pth') 
    else:
        # arch['epoch'] = False
        arch['model_file'] = os.path.join(arch['model_path'], arch['dataset'] + '_' + \
            arch['encode_method'] + '_' + arch['classify_circ'] + '_model.pth')
    
    print('Loading model: ' + os.path.basename(arch['model_file']))
    model = torch.load(arch['model_file']).to(device)

    return model

def model_to_circ(model, x):
    bsz = x.shape[0]

    qdev = tq.QuantumDevice(n_wires=model.n_wires, bsz=bsz, device='cpu', record_op=True)

    # import pdb; pdb.set_trace()
    if hasattr(model, 'encoder'):
        model.encoder(qdev, x)
        op_history_parameterized = qdev.op_history

        qdev.reset_op_history()
        # import pdb; pdb.set_trace()
        encoder_circs = op_history2qiskit_expand_params_new(model.n_wires, op_history_parameterized, bsz=bsz)
    else:
        encoder_circs = []
        for i in range(bsz):
            empty_circ = QuantumCircuit(model.n_wires)
            encoder_circs.append(empty_circ)

    # create the qiskit circuit for q_layer
    model.q_layer(qdev)
    op_history_fixed = qdev.op_history
    qdev.reset_op_history()
    q_layer_circ = op_history2qiskit(model.n_wires, op_history_fixed)

    # create the qiskit circuit for measure
    measurement_circ = tq2qiskit_measurement(qdev, model.measure)
    measure_states = qdev.states 

    # import pdb; pdb.set_trace()
    # assemble the encoder*bsz, trainble quantum layers, and measurement circuits
    assembeld_circs = qiskit_assemble_circs_new(
        encoder_circs, q_layer_circ, measurement_circ)
    
    return assembeld_circs[0] # single qiskit ciruict

def load_qasm_circ(filepath):
    circ = QuantumCircuit.from_qasm_file(filepath)

    return circ

def count_gates_num(circ):
    single_gates, multi_gates = 0, 0
    for instruction in circ.data:
        if instruction.operation.num_qubits == 1:
            single_gates += 1
        elif instruction.operation.num_qubits == 2:
            multi_gates += 1
    print('single_gates: ', single_gates, 'multi_gates: ', multi_gates)

    return single_gates, multi_gates

def save_qasm_file(arch, circ):
    """ save qasm files"""
    if not os.path.exists(arch['qasm_file_path']):
        os.makedirs(arch['qasm_file_path'])
    qasm_file = os.path.join(arch['qasm_file_path'], str(arch['enc_layer'])+'_'+str(arch['classify_circ']) + '.qasm')
    circ.qasm(filename=qasm_file)


if __name__ == '__main__':
    set_seed(42) 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # root_dir = './data_run_seed_42_epoch40_0422/enc_b2_cls_b1'
    root_dir = './quantum_encoding/data_0427'
    # test_states_dir = 'encoded_states_test'

    arch = {'n_wires': 4, 'n_blocks': 2, \
        'use_qiskit': False, \
        'res_root':'./quantum_encoding/0_train_results/', 
        # 'datasets': ["mnist", "fashion", "cifar10", "imdb", "reuters"],
        'datasets': ["mnist"],
        # 'encoded_state_dir_list': ['encoded_states_epoch_20', 'encoded_states_epoch_40', 'encoded_states_epoch_60', 'encoded_states_epoch_80', 'encoded_states_epoch_100', 'encoded_states'],
        'model_list' :  classify_circ_dict,
        # 'enc_layers': ['empty', 'amplitude', 'state_new', 'phase', 'multiphase', 'general', 'multiphase_z', 'IQP_X', 'IQP_Y', 'IQP'], # empty need first
        'enc_layers': ['empty', 'phase', 'multiphase', 'general', 'multiphase_z', 'IQP_X', 'IQP_Y', 'IQP'],
        'valid_model_epochs': False,
        # 'enc_blocks':1, 'cls_blocks':2
        }
    
    datasets_num, layers_num, models_num = len(arch['datasets']), \
        len(arch['enc_layers'])-1, len(arch['model_list']) # metrics_methods without empty
    # for dataset in arch['datasets']:
    Acc, PVI = [], []
    model_mutual_info_enc, model_mutual_info_measure = [], []
    model_mutual_info_renyi_enc, model_mutual_info_renyi_measure = [], []
    renyi_distance_info_enc, renyi_distance_info_measure = [], []
    expressibility_info_enc, expressibility_info_measure = [], []
    entangling_info_enc, entangling_info_measure = [], []

    arch['dataset'] = 'mnist'
    for enc_layer in tqdm(arch['enc_layers'], desc='encode_method', total=len(arch['enc_layers'])):
        arch['enc_layer'] = enc_layer

        for cls_circ in arch['model_list']:
            arch['classify_circ'] = cls_circ
            arch['enc_layer'] = enc_layer
            arch['data_type'] = "image"
            x = torch.randn(2, 32).to(device)
            arch['encode_method'] = "4x8_ryzxy" if enc_layer == "general" else ''

            model_dir = os.path.join(root_dir, arch['dataset'] + '_'+ \
                                    str(arch['enc_layer'])+'_'+str(cls_circ))
            arch['model_path'] =os.path.join(model_dir, 'model')

            print('Loading model:' + arch['model_path'])
            qasm_path = './qasm_files'
            model = load_model(arch)
            circ = model_to_circ(model, x)
            arch['qasm_file_path'] = os.path.join(qasm_path, os.path.basename(root_dir))
            
            save_qasm_file(arch, circ)
            # orig_single, orig_multi = count_gates_num(circ)

                
