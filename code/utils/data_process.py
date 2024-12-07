import os
import sys 
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

# sys.path.append('..')
from models.classify_circ import classify_circ_dict
from baselines.eval_baseline import compute_rank_consistency
from utils.util import load_total_baselines_data
from baselines.pvi import calculate_PVI, visualize_PVI


def cal_score(data, result_path, dataset_name):
    """
        data.shape: models, metrics, encodes
    """
    data_consistency = data

    models_name = ['Model ' + str(i+1) for i in range(data.shape[0])] # models
    metrics_name = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data_consistency.shape[-1] # encodes
    # score.shape: metrics(除开acc) * models
    score = np.zeros((data_consistency.shape[1]-1, data_consistency.shape[0]))
    for model_num in range(data_consistency.shape[0]):

        for metric_num in range(1, data_consistency.shape[1]):
            current_acc = data_consistency[model_num][0]

            for encode_num in range(data_consistency.shape[2]):
                score[metric_num-1][model_num] = \
                    compute_rank_consistency(current_acc, data_consistency[model_num][metric_num])

    np.save(os.path.join(result_path, \
                        'consistency_score_' + dataset_name + '.npy'), score)

    score_pd = DataFrame(score, columns=models_name, index=metrics_name[1:6])
    score_pd.to_csv(os.path.join(result_path, \
                    'consistency_score_' + dataset_name + '.csv'), index=True)

    return score

def cal_origin4_score(data, result_path, dataset_name):
    """
        data.shape: models, metrics, encodes
        just 4 encodes: ['amplitude', 'state_new', 'phase', 'general', 'multiphase', 'multiphase_z', 'IQP', 'IQP_X', "IQP_Y"]
    """
    data_consistency = data[:, :, [0, 2, 3, 4]]

    models_name = ['Model ' + str(i+1) for i in range(data.shape[0])] # models
    metrics_name = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data_consistency.shape[-1] # encodes
    # score.shape: metrics(除开acc) * models
    score = np.zeros((data_consistency.shape[1]-1, data_consistency.shape[0]))
    for model_num in range(data_consistency.shape[0]):

        for metric_num in range(1, data_consistency.shape[1]):
            current_acc = data_consistency[model_num][0]

            for encode_num in range(data_consistency.shape[2]):
                score[metric_num-1][model_num] = \
                    compute_rank_consistency(current_acc, data_consistency[model_num][metric_num])

    np.save(os.path.join(result_path, \
                        'consistency_orig4_score_' + dataset_name + '.npy'), score)

    score_pd = DataFrame(score, columns=models_name, index=metrics_name[1:6])
    score_pd.to_csv(os.path.join(result_path, \
                    'consistency_orig4_score_' + dataset_name + '.csv'), index=True)

    return score

def cal_base6_score(data, result_path, dataset_name):
    """
        data.shape: models, metrics, encodes
        total encodes: ['amplitude', 'state_new', 'phase', 'general', 'multiphase', 'multiphase_z', 'IQP', 'IQP_X', "IQP_Y"]
        just 5 sub_encodes: ['amplitude', 'state_new', 'phase', 'general', 'IQP']
    """
    data_consistency = data

    models_name = ['Model ' + str(i+1) for i in range(data.shape[0])] # models
    metrics_name = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data_consistency.shape[-1] # encodes
    # score.shape: metrics(除开acc) * models
    score = np.zeros((data_consistency.shape[1]-1, data_consistency.shape[0]))
    for model_num in range(data_consistency.shape[0]):

        for metric_num in range(1, data_consistency.shape[1]):
            current_acc = data_consistency[model_num][0]

            for encode_num in range(data_consistency.shape[2]):
                score[metric_num-1][model_num] = \
                    compute_rank_consistency(current_acc, data_consistency[model_num][metric_num])

    np.save(os.path.join(result_path, \
                        'consistency_orig4_score_' + dataset_name + '.npy'), score)

    score_pd = DataFrame(score, columns=models_name, index=metrics_name[1:6])
    score_pd.to_csv(os.path.join(result_path, \
                    'consistency_orig4_score_' + dataset_name + '.csv'), index=True)

    return score


def single_accuracy_consistency(data, result_path):
    """ evaluate consistency of baselines order """
    # data.shape: metrics, encodes, models
    data_consistency = np.swapaxes(data, 1, 2)
    data_consistency = np.swapaxes(data_consistency, 0, 1) # models, metrics, encodes

    # score.shape: metrics * models
    score = np.zeros((data_consistency.shape[1]-1, data_consistency.shape[0]))
    for model_num in range(data_consistency.shape[0]):

        for metric_num in range(1, data_consistency.shape[1]):
            current_acc = data_consistency[model_num][0]

            for encode_num in range(data_consistency.shape[2]):
                score[metric_num-1][model_num] = \
                    compute_rank_consistency(current_acc, data_consistency[model_num][metric_num])

    np.save(os.path.join(result_path, 'score.npy'), score)
    models_name = ['Model ' + str(i+1) for i in range(data.shape[-1])] # models
    metrics_name = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data.shape[0] # metrics

    score_pd = DataFrame(score, columns=models_name, index=metrics_name[1:6])
    score_pd.to_csv(os.path.join(result_path, 'score.csv'), index=True)


def multi_accuracy_consistency(merge_data_path, result_path):
    data = load_total_baselines_data(merge_data_path) # data.shape: metrics, datasets, encodes, models
    data_consistency = np.transpose(data, (1, 3, 0, 2)) # datasets, models, metrics, encodes

    datasets = ["mnist", "fashion", "cifar10", "imdb", "reuters"]
    models_name = ['Model ' + str(i+1) for i in range(data.shape[-1])] # models
    metrics_name = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data.shape[0]
    # score.shape: datasets * metrics(除开acc) * models
    score = np.zeros((data_consistency.shape[0], data_consistency.shape[2]-1, data_consistency.shape[1]))
    for dataset in range(data_consistency.shape[0]):
        for model_num in range(data_consistency.shape[1]):

            for metric_num in range(1, data_consistency.shape[2]):
                current_acc = data_consistency[dataset][model_num][0]

                for encode_num in range(data_consistency.shape[3]):
                    score[dataset][metric_num-1][model_num] = \
                        compute_rank_consistency(current_acc, data_consistency[dataset][model_num][metric_num])

        np.save(os.path.join(merge_data_path, \
                            'consistency_score_' + datasets[dataset] + '.npy'), score[dataset])

        score_pd = DataFrame(score[dataset], columns=models_name, index=metrics_name[1:6])
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        csv_path = os.path.join(result_path, 'consistency_score_' + datasets[dataset] + '.csv')
        score_pd.to_csv(csv_path, index=True)
        print('save consistency score of dataset: ', datasets[dataset])
    
    return score

if __name__ == '__main__':
    arch = {'n_wires': 4, 'n_blocks': 2, \
            'use_qiskit': False, \
            'res_root':'./quantum_code/quantum_encoding/',
            'datasets': ['mnist',],
            'encoded_state_dir_list': ['encoded_states_epoch_40', 'encoded_states'],
            'enc_layers': ['general', 'multiphase', 'phase', 'state', 'empty'],
            'classify_circ': ['model2', 'model3'],
            'train_model_dir': 'data_run_seed_42_epoch30_encode',
            'valid_model_epochs': True}

    for encoded_state_dir in arch['encoded_state_dir_list']:
        arch['encoded_state_dir'] = encoded_state_dir
        arch['current_result_dir'] = os.path.join(
                                        arch['res_root'], encoded_state_dir)
        if not os.path.exists(arch['current_result_dir']):
            os.makedirs(arch['current_result_dir'], exist_ok=True)
        calculate_PVI(arch)
        pvi_res, cal = visualize_PVI(arch)
        np.save(arch['current_result_dir']+'/0_pvi_res.npy', np.array(pvi_res))
        np.save(arch['current_result_dir']+'/0_pvi_cal.npy', np.array(cal))
