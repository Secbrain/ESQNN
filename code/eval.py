import os
import torch
import time
import numpy as np
from tqdm import tqdm

from utils.util import set_seed, load_model_data
from models.classify_circ import classify_circ_dict
from baselines.mutual_info import cal_mutual_info, cal_mutual_info_renyi
from baselines.renyi_divergence import cal_renyi_divergence
from baselines.pqc_capabilities import calculate_expressibility, calculate_entangling

def eval_single_model_baselines(arch, root_dir, result_path, single_states_dir):
    datasets_num = len(arch['datasets'])
    models_num = len(arch['model_list'])
    layers_num = len(arch['enc_layers'])

    Acc, PVI = [], []
    model_mutual_info_measure = []
    model_mutual_info_renyi_measure = []
    renyi_distance_info_measure = []
    expressibility_info_measure = []
    entangling_info_measure = []

    for dataset in arch['datasets']:
        arch['dataset'] = dataset
        for cls_circ in arch['model_list']:
            for enc_layer in tqdm(arch['enc_layers'], desc='Encoding Method', total=len(arch['enc_layers'])):
                arch['enc_layer'] = enc_layer
                arch['classify_circ'] = cls_circ

                if dataset == "vowel":
                    arch['data_type'] = "vowel"
                elif dataset in ['imdb', "reuters"]:
                    arch['data_type'] = "nlp"
                else:
                    arch['data_type'] = "image"

                model_dir = os.path.join(root_dir, f"{arch['dataset']}_{arch['enc_layer']}_{cls_circ}")
                arch['model_path'] = os.path.join(model_dir, 'model')
                arch['encoded_states_dir'] = os.path.join(model_dir, single_states_dir)
                arch['empty_model_dir'] = os.path.join(root_dir, f"{arch['dataset']}_empty_{arch['enc_layer']}_{cls_circ}")

                if enc_layer == "general":
                    if arch['enc_blocks'] == 2:
                        arch['encode_method'] = "4x8_ryzxy"
                    elif arch['enc_blocks'] == 1:
                        arch['encode_method'] = "4x4_ryzxy"
                else:
                    arch['encode_method'] = ''
                empty_out_log = np.load(os.path.join(arch['empty_model_dir'], single_states_dir,
                                                       f"{arch['dataset']}_{enc_layer}_out_log.npy"))
                
                enc_data, measure_data, pre_enc_data, out_log_data, gt_label = load_model_data(arch)
                print(f'Running eval_model: {enc_layer}, Model: {cls_circ}')
                eval_model(arch)

                with open(os.path.join(result_path, 'eval_baselines.txt'), 'a') as f:
                    print(f'Running eval_baselines: {arch['enc_layer']}, Model: {cls_circ}', file=f)

                    if 'Acc' in arch['eval_baselines']:
                        predict_label = np.argmax(out_log_data, axis=1)
                        acc = np.sum(predict_label == gt_label) / len(gt_label)
                        Acc.append(acc)
                        pvi_score = np.zeros([len(gt_label), 2])
                        if 'PVI' in arch['eval_baselines']:
                            for num in range(len(gt_label)):
                                pvi_score[num, 0] = -np.log2(empty_out_log[num, int(gt_label[num])])
                                pvi_score[num, 1] = -np.log2(out_log_data[num, int(gt_label[num])])
                            pvi = np.mean(pvi_score[:, 0] - pvi_score[:, 1])
                            print(f'Acc: {acc} \t PVI: {pvi}')
                            f.write(f'Acc: {acc} \t PVI: {pvi}\n')
                            PVI.append(pvi)

                    if 'mutual_info' in arch['eval_baselines']:
                        mutual_measure = cal_mutual_info(measure_data, pre_enc_data)
                        model_mutual_info_measure.append(mutual_measure)
                        print(f'Mutual Info: {mutual_measure}', file=f)
                        f.write(f'Mutual Info: {mutual_measure}\n')

                    if 'mutual_info_renyi' in arch['eval_baselines']:
                        mutual_renyi_measure = cal_mutual_info_renyi(measure_data, pre_enc_data)
                        model_mutual_info_renyi_measure.append(mutual_renyi_measure)
                        print(f'Renyi Mutual Info: {mutual_renyi_measure}', file=f)
                        f.write(f'Renyi Mutual Info: {mutual_renyi_measure}\n')

                    if 'renyi_distance' in arch['eval_baselines']:
                        renyi_divergence = cal_renyi_divergence(measure_data)
                        renyi_distance_info_measure.append(renyi_divergence)
                        print(f'Renyi Distance: {renyi_divergence}', file=f)
                        f.write(f'Renyi Distance: {renyi_divergence}\n')

                    if 'exprss_entangle' in arch['eval_baselines']:
                        expre_measure = calculate_expressibility(measure_data)
                        expressibility_info_measure.append(expre_measure)
                        entangle_measure = calculate_entangling(measure_data)
                        entangling_info_measure.append(entangle_measure)
                        print(f'Expressibility: {expre_measure} \t Entangling: {entangle_measure}', file=f)
                        f.write(f'Expressibility: {expre_measure} \t Entangling: {entangle_measure}\n')

                    print('*'*40, f"{arch['dataset']} eval_baselines END", '*'*40, file=f)
                    f.write('\n\n')

    date = time.strftime("%m%d", time.localtime())

    if 'Acc' in arch['eval_baselines']:
        acc_data = np.array(Acc).reshape((datasets_num, models_num, layers_num))
        np.save(os.path.join(result_path, f"{arch['dataset']}_acc.npy"), acc_data)
        if 'PVI' in arch['eval_baselines']:
            pvi_data = np.array(PVI).reshape((datasets_num, models_num, layers_num))
            np.save(os.path.join(result_path, f"{arch['dataset']}_pvi.npy"), pvi_data)

    if 'mutual_info' in arch['eval_baselines']:
        model_mutual_info_measure = np.array(model_mutual_info_measure).reshape((datasets_num, models_num, layers_num))
        np.save(os.path.join(result_path, f"{arch['dataset']}_mutual_info_measure.npy"), model_mutual_info_measure)

    if 'mutual_info_renyi' in arch['eval_baselines']:
        model_mutual_info_renyi_measure = np.array(model_mutual_info_renyi_measure).reshape((datasets_num, models_num, layers_num))
        np.save(os.path.join(result_path, f"{arch['dataset']}_mutual_info_renyi_measure.npy"), model_mutual_info_renyi_measure)

    if 'renyi_distance' in arch['eval_baselines']:
        renyi_distance_info_measure = np.array(renyi_distance_info_measure).reshape((datasets_num, models_num, layers_num))
        np.save(os.path.join(result_path, f"{arch['dataset']}_renyi_distance_measure.npy"), renyi_distance_info_measure)

    if 'exprss_entangle' in arch['eval_baselines']:
        expressibility_info_measure = np.array(expressibility_info_measure).reshape((datasets_num, models_num, layers_num))
        np.save(os.path.join(result_path, f"{arch['dataset']}_expressibility_measure.npy"), expressibility_info_measure)
        entangling_info_measure = np.array(entangling_info_measure).reshape((datasets_num, models_num, layers_num))
        np.save(os.path.join(result_path, f"{arch['dataset']}_entangling_measure.npy"), entangling_info_measure)

if __name__ == '__main__':
    set_seed(42)  # Set random seed
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    arch = {
        'n_wires': 4, 'n_blocks': 2,
        'use_qiskit': False,
        'datasets': ["mnist"],
        'model_list': classify_circ_dict,  # Full set of test models
        'enc_layers': ['amplitude', 'state', 'phase', 'general', 'multiphase', 'multiphase_z', 'IQP', 'IQP_X', "IQP_Y"],
        'eval_baselines': ['Acc', 'mutual_info', 'renyi_distance', 'expressibility_entangling', 'PVI'],
        'valid_model_epochs': False,
        'enc_blocks': 1, 'cls_blocks': 2,
    }
    root_dir = '../quantum_encoding_data/train_models/mnist_0508_state/enc_b1_cls_b1'
    result_path = '../quantum_encoding_data/1_output/baselines/mnist_0508_e1m1'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if arch['valid_model_epochs']:
        epochs_list = np.arange(0, 31, 5)
        epochs_list[0] = 1
        test_states_dir = [
            'encoded_states_test_epoch'+str(epoch) for epoch in epochs_list
        ]
    else:
        test_states_dir = 'encoded_states_test'  # Adjust the corresponding dataset part in valid_model as well

    # Remove 'amplitude' and 'state_new' if using more than one encoding block
    if arch['enc_blocks'] > 1:
        if 'amplitude' in arch['enc_layers']:
            arch['enc_layers'].remove('amplitude')
        if 'state_new' in arch['enc_layers']:
            arch['enc_layers'].remove('state_new')

    # Evaluate models for each epoch if a list of epochs is provided
    if isinstance(test_states_dir, list):
        for single_states_dir in test_states_dir:
            result_path_with_epoch = os.path.join(result_path, single_states_dir)
            eval_single_model_baselines(
                arch, root_dir, result_path_with_epoch, single_states_dir
            )
    # Evaluate the model for the default test state directory
    elif isinstance(test_states_dir, str):
        eval_single_model_baselines(arch, root_dir, result_path, test_states_dir)