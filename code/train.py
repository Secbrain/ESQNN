import os
import sys
import random
import torch
import yaml
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.entire_circ import TempleteCirc
from models.classify_circ import classify_circ_dict
from utils.util import set_seed, get_dataset, valid_model, valid_test_model, load_dataset

def train(dataflow, model, device, optimizer, arch):
    criterion = nn.CrossEntropyLoss()
    for feed_dict in dataflow['train']:
        if arch['data_type'] == 'nlp':
            inputs = feed_dict[0].to(device)
            targets = feed_dict[1].type(torch.LongTensor).to(device)
        else:
            inputs = feed_dict[arch['data_type']].to(device)
            targets = feed_dict['digit'].to(device)
        
        if arch['train_empty']:
            inputs = torch.zeros_like(inputs)
        outputs, bsz_pre_enc, bsz_enc_data, out_pre_log = model(inputs, arch)
        
        # print('pre_enc: ', bsz_pre_enc, 'bsz_enc_data: ', bsz_enc_data, 'outputs: ', outputs)
        # loss = F.nll_loss(outputs, targets)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def valid_test(dataflow, split, model, device, arch, qiskit=False):
    target_all = []
    output_all = []
    
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            if arch['data_type'] == 'nlp':
                inputs = feed_dict[0].to(device)
                targets = feed_dict[1].type(torch.LongTensor).to(device)
            else:
                inputs = feed_dict[arch['data_type']].to(device)
                targets = feed_dict['digit'].to(device)

            out, bsz_pre_enc, bsz_enc_data, out_pre_log = model(inputs, arch, use_qiskit=qiskit)
            # print('pre_enc: ', bsz_pre_enc, 'bsz_enc_data: ', bsz_enc_data, 'outputs: ', out)

            target_all.append(targets)
            output_all.append(out)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")
    
    return accuracy, loss


def run_qiskit(model, dataflow, device):
    try:
        from qiskit import IBMQ
        from torchquantum.plugins import QiskitProcessor

        # firstly perform simulate
        processor_simulation = QiskitProcessor(use_real_qc=False)
        model.set_qiskit_processor(processor_simulation)
        valid_test(dataflow, 'test', model, device, qiskit=True)

        # then try to run on REAL QC
        backend_name = 'ibmq_lima'
        print(f"Test on Real Quantum Computer {backend_name}")
        # Please specify your own hub group and project if you have the
        # IBMQ premium plan to access more machines.
        processor_real_qc = QiskitProcessor(use_real_qc=True,
                                            backend_name=backend_name,
                                            hub='ibm-q',
                                            group='open',
                                            project='main',
                                            )
        model.set_qiskit_processor(processor_real_qc)
        valid_test(dataflow, 'test', model, device, qiskit=True)
    except ImportError:
        print("Please install qiskit, create an IBM Q Experience Account and "
            "save the account token according to the instruction at "
            "'https://github.com/Qiskit/qiskit-ibmq-provider', "
            "then try again.")
    except :
        print(f"Can not test with Qiskit Simulator")
        

# def run(arch):
if __name__ == '__main__':
    # eliminate GPU training randomness
    cudnn.benchmark = False
    cudnn.deterministic = True
    # cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    args, opts = parser.parse_known_args()

    with open(args.config, 'r') as f:
        arch = yaml.safe_load(f)

    set_seed(42)
    dataflow = load_dataset(arch)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.exists(arch['filepath'] + 'model/'):
        os.makedirs(arch['filepath'] + 'model/', exist_ok=True)
    
    # set model
    model = TempleteCirc(arch).to(device)
    n_epochs = arch['epochs']
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    if arch['static']:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=arch['wires_per_block'])

    arch['model_file'] = arch['filepath'] + 'model/' + arch['dataset'] + '_' + \
            arch['encode_method'] + '_' + arch['classify_circ'] + '_model.pth'
    arch['encoded_states_dir'] = os.path.join(arch['filepath'], \
                                                arch['encoded_states_dir_name'])
    tmp_acc = -np.inf
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer, arch)

        # valid
        accuracy, loss = valid_test(dataflow, 'valid', model, device, arch)
        scheduler.step()

        # test
        test_accuracy, test_loss = valid_test(dataflow, 'test', model, device, 
                                            arch, qiskit=False)
        if tmp_acc < test_accuracy:
            print('epoch: ', epoch, 'tmp_acc: ', tmp_acc, 'test_accuracy: ', test_accuracy)
            torch.save(model, arch['model_file']) 
            tmp_acc = test_accuracy

        if epoch % 1 == 0 and arch['save_epoch']:
            epoch_file =arch['filepath'] + 'model/' + arch['dataset'] + '_' + \
                        arch['encode_method'] + '_' + arch['classify_circ'] + \
                        '_epoch' + str(epoch) +'.pth'
            torch.save(model, epoch_file)
        
        """ 测试每个epoch的性能 """
        ''' test_accuracy, test_loss = valid_test(dataflow, 'test', model, device, 
                                        arch, qiskit=False)
        with open(arch['train_res_file'], 'a+') as file:
            file.write(str(enc_layer)+ '\t'+ str(arch['encode_method']) + '\t' + \
                    arch['dataset'] +'\t' + str(epoch) + '\t' + \
                    str(accuracy) + '\t' + \
                    str(loss) + '\t' + str(test_accuracy) + '\t' + str(test_loss) + '\n') '''
            
    model_type = 'empty' if arch['train_empty'] else 'templete'
    if arch['train_empty']:
        arch['train_res_file'] = arch['train_res_file'].replace('.txt', '_empty.txt')
    with open(arch['train_res_file'], 'a+') as file:
        file.write(str(arch['enc_layer'])+ '\t'+ str(arch['encode_method']) + '\t' + \
                arch['dataset'] +'\t' + str(model_type) + '\t'+str(epoch) + '\t' + \
                arch['classify_circ'] + '\t' + str(accuracy) + '\t' + \
                str(loss) + '\t' + str(test_accuracy) + '\t' + str(test_loss) + '\n')
    
    if arch['train_empty']:
        print('last model acc:', test_accuracy)
        torch.save(model, arch['model_file'].replace('.pth', '_last.pth')) 
    
    # save model and encoded states
    valid_test_model(arch, device, model, dataflow)
    # torch.save(model.to('cpu'), arch['model_file']) 

    if arch['use_qiskit']:
        run_qiskit(model, dataflow, device)

