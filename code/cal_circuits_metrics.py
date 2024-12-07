import os
import torch
import time
import sys
sys.path.append('./quantum_encode_evaluation')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from utils.util import set_seed, load_model_data
from models.classify_circ import classify_circ_dict
from baselines.renyi_divergence import cal_renyi_divergence
from baselines.pqc_capabilities import calculate_expressibility, calculate_expressibility, calculate_entangling
from baselines.von_entropy import quantum_von_neumann_p

data_dir = './data/encoded_states/'
save_data_dir = './data/paper_data/'

""" The four information indexes corresponding to the four codes are calculated """
def cal_encode_metrics():
    datasets_dir = ['mnist_4_enc1_blk2', 'fashion_4_enc1_blk2']
    arch= {'eval_baselines' : ['mutual_info', 'renyi_distance', 'exprss', 'entangle']}  # four metrics
    model_lists = ['model'+str(i) for i in range(1, 20)]
    # arch= {'eval_baselines' : ['mutual_info']}
    for dataset_dir in datasets_dir:
        cal_metrics = np.zeros((4, 4))
        for enc_num, current_encode in enumerate(['amplitude', 'state_new', 'multiphase_z', "IQP_Y"]):
            enc_data_dir = os.path.join(data_dir, dataset_dir, current_encode, 'model1') 
            measure_data = np.load(os.path.join(enc_data_dir, 'test_enc_data.npy'))
            # pre_enc_data = np.load(os.path.join(enc_data_dir, 'test_pre_enc.npy'))

            print('\n\n', 'test sample numbers:', measure_data.shape[0])

            if 'mutual_info' in arch['eval_baselines']:
                # MI renyi
                start_time = time.time()
                mutual_measure = quantum_von_neumann_p(measure_data)
                print('mutual_info:', mutual_measure, \
                    'execute time', np.round(time.time() - start_time, 2))
            if 'renyi_distance' in arch['eval_baselines']:
                # QSD
                start_time = time.time()
                renyi_measure = cal_renyi_divergence(measure_data)
                print('renyi_distance:', renyi_measure, \
                'execute time', np.round(time.time() - start_time, 2))
            if 'exprss' in arch['eval_baselines']:
                # Expressibility
                start_time = time.time()
                exprss_entangle = calculate_expressibility(measure_data)
                print('exprss:', exprss_entangle , \
                    'execute time', np.round(time.time() - start_time, 2))
            if 'entangle' in arch['eval_baselines']:
                # entangle
                start_time = time.time()
                entangle = calculate_entangling(measure_data)
                print('entangle:', entangle,\
                    'execute time', np.round(time.time() - start_time, 2))
            cal_metrics[enc_num] = [mutual_measure, renyi_measure, exprss_entangle, entangle]
            # cal_metrics[enc_num] = [mutual_measure, 0, 0, 0]

        data = pd.DataFrame(cal_metrics)
        data.columns =['mutual_info', 'renyi_distance', 'exprss', 'entangle']
        data.index = ['amplitude', 'state_new', 'multiphase_z', "IQP_Y"]
        data.to_csv(os.path.join(save_data_dir, 'metrics', dataset_dir+'_metrics_encode.csv'))

def cal_circuits_metrics():
    """ Calculate each encode corresponding to 19 circuits of the four metrics
        the calculation of 'code +  training PQC' four indicators of information"""
    datasets_dir = ['mnist_4_enc1_blk2', 'fashion_4_enc1_blk2']
    arch= {'eval_baselines' : ['mutual_info', 'renyi_distance', 'exprss', 'entangle']} # four metrics
    model_lists = ['model'+str(i) for i in range(1, 20)]
    for dataset_dir in datasets_dir:
        for current_encode in ['amplitude', 'state_new', 'multiphase_z', "IQP_Y"]:
            # for enc_num, current_encode in enumerate(['e1']):
            cal_metrics = np.zeros((19, 4))
            for model_num, model_dir in enumerate(model_lists):
                enc_data_dir = os.path.join(data_dir, dataset_dir, current_encode, model_dir)
                measure_data = np.load(os.path.join(enc_data_dir, 'test_measure_data.npy'))
                # pre_enc_data = np.load(os.path.join(enc_data_dir, 'test_pre_enc.npy'))

                print('\n\n', 'test sample numbers:', measure_data.shape[0])

                if 'mutual_info' in arch['eval_baselines']:
                    # MI renyi
                    start_time = time.time()
                    mutual_measure = quantum_von_neumann_p(measure_data)
                    print('mutual_info:', mutual_measure, \
                        'execute time', np.round(time.time() - start_time, 2))
                if 'renyi_distance' in arch['eval_baselines']:
                    # QSD
                    start_time = time.time()
                    renyi_measure = cal_renyi_divergence(measure_data)
                    print('renyi_distance:', renyi_measure, \
                    'execute time', np.round(time.time() - start_time, 2))
                if 'exprss' in arch['eval_baselines']:
                    # Expressibility
                    start_time = time.time()
                    exprss_entangle = calculate_expressibility(measure_data)
                    print('exprss:', exprss_entangle , \
                        'execute time', np.round(time.time() - start_time, 2))
                if 'entangle' in arch['eval_baselines']:
                    # entangle
                    start_time = time.time()
                    entangle = calculate_entangling(measure_data)
                    print('entangle:', entangle,\
                        'execute time', np.round(time.time() - start_time, 2))
                cal_metrics[model_num] = [mutual_measure, renyi_measure, exprss_entangle, entangle]
                # cal_metrics[model_num] = [mutual_measure, 0, 0, 0]

            data = pd.DataFrame(cal_metrics)
            data.columns =['mutual_info', 'renyi_distance', 'exprss', 'entangle']
            data.insert(0, 'Circuit', model_lists)
            data.to_csv(os.path.join(save_data_dir, 'metrics', dataset_dir+'_'+current_encode+'_circuits_metrics.csv'))
    # np.save('./data/mnist/cal_metrics.npy', cal_metrics)

def refine_metrics():
    encode_lists = ['IQP_Y']
    # encode_lists = ['amplitude', 'state_new', 'multiphase_z', "IQP_Y"]
    enc_dir = 'fashion_4_enc1_blk2'
    df = pd.read_csv('./data/paper_data/'+str(enc_dir)+'_entropy.csv', header=[0, 1])
    for encode in encode_lists:
        data_path = './data/paper_data/metrics/'+str(enc_dir)+'_'+str(encode)+'_circuits_metrics.csv'
        data = pd.read_csv(data_path)
        data['renyi_distance'] = data['renyi_distance'] / 100
        data['exprss'] = data['exprss'] / 100
        data['mutual_info'] = df[(str(encode), 'Entropy')].values / 1e4
        data.to_csv('./data/paper_data/refine_metrics/'+str(enc_dir)+'_'+str(encode)+'_circuits_metrics.csv', index=False)

def com_river_data():
    """ 合并ACC生成 river data"""
    # dataset_dir = ['mnist_4_enc1_blk2', 'fashion_4_enc1_blk2']
    dataset_dir = ['fashion_4_enc1_blk2']
    # encode_list = ['amplitude', 'state_new', 'multiphase_z', "IQP_Y"]
    encode_list = ['multiphase_z', "IQP_Y"]
    for dataset in dataset_dir:
        for encode in encode_list:
            print('dataset: ', dataset, 'encode: ', encode)
            acc = pd.read_csv('./data/paper_data/'+str(dataset).replace('_enc1', '')+'_base.csv')
            data = pd.read_csv('./data/paper_data/refine_metrics/'+str(dataset)+'_'+str(encode)+'_circuits_metrics.csv')
            # data = data.drop(['Unnamed: 0.1'], axis=1) # 如果有多余的index列，删掉
            data.index = data.index + 1
            data.insert(1, 'ACC', acc[encode][1:])
            data.columns = ['Circuit', 'ACC', 'Entropy', 'Renyi Divergence', 'Expressibility', 'Entanglement']
            data.to_csv('./data/paper_data/river_data/'+str(dataset)+'_'+str(encode)+'_river_data.csv', index=False)

cal_encode_metrics()
cal_circuits_metrics()