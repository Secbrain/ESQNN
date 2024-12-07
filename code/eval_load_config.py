""" noisy test"""
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', \
                        default='./configs/default_eval.yml', help='config file')
    args, opts = parser.parse_known_args()

    noise_json_root = './cali_data/IBMQ_calibration_data_2021'
    arch = {'n_wires': 4, \
            'static': False,
            'n_epochs': 30,
            # 'n_epochs': 1,
            'save_test_encode_data': True, 
            'train_empty': False,
            'backend_list': ['ibm_brisbane', 'ibm_kyoto', 'ibm_osaka'],
            'noise_test': True,
            'noise_factor': 1, 
            'enc_layer': 'amplitude', 'use_qiskit': False, \
            'filepath': './data',
            'valid_model_epochs': True,
            'save_epoch': False,
            'data_type': 'image',
            'dataset': 'fashion',
            # 'dataset': 'mnist',
            'datasets_list': ['mnist'],
            'encoding_method': {'e1':'state_new', 'e2':'amplitude', 
                   'e3':'multiphase_z', 'e4':'IQP_Y'},
            'enc_layer_list': ['amplitude', 'state_new', 'multiphase_z', "IQP_Y"],
            'classify_circ_list': ['model'+str(i) for i in range(1,20)], 
            'class_num': 4,
            "rotation_gate": "rz", # toward IQP encoding
            'enc_blocks':1, 'cls_blocks':2, 
            'encoded_states_dir_name': 'encoded_states',
            'classify_circ': 'model6', 
            }
    arch['config_file'] = args.config

    # circuit block
    blocks_list = [1, 3, 4, 5]
    for cls_blocks in blocks_list:
        arch['cls_blocks'] = cls_blocks
    
        if arch['noise_test']:
            # noisy test
            for backend in arch['backend_list']:
                arch['noise_category'] = 'gate_and_read'
                arch['backend_name'] = backend
                # dataset
                for dataset in arch['datasets_list']:
                    print('dataset: ', dataset)
                    # circuit block
                    for enc_layer in arch['enc_layer_list']:
                        arch['enc_layer'] = enc_layer

                        # trainable circuit
                        for classify_circ in arch['classify_circ_list']:
                            arch['classify_circ'] = classify_circ
                            arch['dataset'] = dataset
                            # with open(arch['config_file_1'], "w") as f:
                            with open(arch['config_file'], "w") as f:
                                json.dump(arch, f)
                            return_code = os.system('python eval.py ' + arch['config_file'])
                            if return_code != 0:
                                print('eval.py error!')
                                exit(1)
        else:
            # dataset
            for dataset in arch['datasets_list']:
                print('dataset: ', dataset)
                # circuit block
                for enc_layer in arch['enc_layer_list']:
                    arch['enc_layer'] = enc_layer

                    # trainable circuit
                    for classify_circ in arch['classify_circ_list']:
                        arch['classify_circ'] = classify_circ
                        arch['dataset'] = dataset
                        # with open(arch['config_file_1'], "w") as f:
                        with open(arch['config_file'], "w") as f:
                            json.dump(arch, f)
                        return_code = os.system('python eval.py ' + arch['config_file'])
                        if return_code != 0:
                            print('eval.py error!')
                            exit(1)