import os
import random
import numpy as np
import torch
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from keras.datasets import imdb, reuters
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences

from torchquantum.datasets import MNIST, CIFAR10, Vowel

__all__ = ['set_seed', 'PCA_reduction', 'filter_data', 'get_dataset',\
           'eval_model', 'save_npy_file', 'valid_model', 'valid_test_model',\
           'save_numpy_data', 'reshape_data', 'load_total_baselines_data',\
           'load_single_dataset_data', 'extend_one_sort', 'save_baselines',\
           'load_model_data', 'load_npy_file', 'merge_data']
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset(arch):
    dataset = get_dataset(arch, arch['dataset'])
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=16,
            pin_memory=True)
    
    return dataflow

def PCA_reduction(sentence_matrix,  maxlen=50, n_components=16):
    # padding sequence, and list to numpy
    sentence_matrix = pad_sequences(sentence_matrix, maxlen=maxlen, padding='post')
    sentence_matrix = np.array([np.array(i) for i in sentence_matrix])

    pca = PCA(n_components=n_components)
    pca.fit(sentence_matrix)
    sentence_matrix_reduced = pca.transform(sentence_matrix)
    return sentence_matrix_reduced

def filter_data(x, y, value1=3, value2=4):
    #  filter value1 data
    indices = np.where(y == value1)[0]
    value2_num = len(np.where(y == value2)[0])
    random_indices = np.random.choice(indices, size=value2_num, replace=False)
    x_value1, y_value1 = x[random_indices], y[random_indices]
    y_value1 = np.zeros_like(y_value1) 

    # filter value2 data
    indices = np.where(y == value2)[0]
    x_value2, y_value2 = x[indices], y[indices]
    y_value2 = np.ones_like(y_value2) 
    
    x_train, y_train = np.concatenate((x_value1, x_value2)), np.concatenate((y_value1, y_value2))
    
    return x_train, y_train

def get_dataset(arch, dataset_name):
    if dataset_name == "mnist":
        if arch['class_num'] == 4:
            return MNIST(
                root='./datasets/mnist_data',
                center_crop=24,
                train_valid_split_ratio=[0.9, 0.1],
                digits_of_interest=[0, 1, 2, 3],
                # n_test_samples=75
            )
        elif arch['class_num'] == 2:
            return MNIST(
                root='./datasets/mnist_data',
                center_crop=24,
                train_valid_split_ratio=[0.9, 0.1],
                digits_of_interest=[3, 6],
                # n_test_samples=75
            )
    elif dataset_name ==  "cifar10":
        return CIFAR10(
            root='./datasets/cifar10_data',
            center_crop=28,
            resize=24,
            grayscale=True,
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=[3, 6],
            # n_test_samples=75
        )
    elif dataset_name == "vowel":
        return Vowel(
            root='./datasets/vowel_data',
            resize=10,
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=[3, 6],
            test_ratio=0.1
        )
    elif dataset_name == "fashion":
        if arch['class_num'] == 4:
            return MNIST(
                root='./datasets/fashion_data',
                center_crop=24,
                fashion=True,
                train_valid_split_ratio=[0.9, 0.1],
                digits_of_interest=[0, 1, 2, 3],
                # n_test_samples=75
            )
        elif arch['class_num'] == 2:
            return MNIST(
                root='./datasets/fashion_data',
                center_crop=24,
                fashion=True,
                train_valid_split_ratio=[0.9, 0.1],
                digits_of_interest=[3, 6],
                # n_test_samples=75
            )
    # NLP Datasets
    elif dataset_name in ["imdb", "reuters"]:
        maxlen = 65
        if dataset_name == "imdb":
            (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=maxlen)
        elif dataset_name == "reuters":
            (x_train, y_train), (x_test, y_test) = reuters.load_data(maxlen=maxlen)
            # filter dataset into category_num(3) == category_num(4)
            x_train, y_train = filter_data(x_train, y_train, value1=3, value2=4)
            x_test, y_test = filter_data(x_test, y_test, value1=3, value2=4)
            
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        # import pdb; pdb.set_trace()

        # pca decompose
        x_train = PCA_reduction(x_train, maxlen=maxlen, n_components=16 * arch['enc_blocks'])
        x_valid = PCA_reduction(x_valid, maxlen=maxlen, n_components=16 * arch['enc_blocks'])
        x_test = PCA_reduction(x_test, maxlen=maxlen, n_components=16 * arch['enc_blocks'])

        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        valid_dataset = TensorDataset(torch.Tensor(x_valid), torch.Tensor(y_valid))
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        dataset = {'train': train_dataset, 'valid':valid_dataset, 'test': test_dataset}

        return dataset
    
    else:
        print("No such dataset")
        return None
    
def eval_model(arch, dataflow):
    model_path = os.path.join(arch['filepath'], 'model/')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if arch['valid_model_epochs']:
        for epoch in range(0, 31, 5):
            epoch = 1 if epoch == 0 else epoch
            arch['epoch'] = epoch
            arch['model_file'] = model_path + arch['dataset'] + '_' + \
                arch['encode_method'] + '_' + arch['classify_circ'] + '_epoch' + str(epoch) + '.pth'
            model = torch.load(arch['model_file']).to(device)
            valid_test_model(arch, device, model, dataflow)
    else:
        
        if arch['model_type'] == 'empty' and arch['test_empty_last']:
            arch['model_file'] = model_path + arch['dataset'] + '_' + \
            arch['encode_method'] + '_' + arch['classify_circ'] + '_model_last.pth' 
        else:
            arch['model_file'] = model_path + arch['dataset'] + '_' + \
                arch['encode_method'] + '_' + arch['classify_circ'] + '_model.pth'
        model = torch.load(arch['model_file']).to(device)
        # valid_test_model(arch, device, model, dataflow)
        valid_test_model_255(arch, device, model, dataflow)

def save_npy_file(arch, input_data, \
                  target_data, pre_data, enc_data, out_data, out_pre_data, measure_data):
    
    if 'epoch' in arch.keys():
        npy_epoch_dir = arch['encoded_states_dir'] + '_epoch_' + \
            str(arch['epoch']) + '/'
        npy_dir = npy_epoch_dir
        print('save npy file to: ', npy_dir)
    else: 
        npy_dir = arch['encoded_states_dir'] + '/'
        print('save npy file to: ', npy_dir)
    if os.path.exists(npy_dir) is False:
            os.makedirs(npy_dir, exist_ok=True)
    # arch['filepath']
    # import pdb; pdb.set_trace()
    npy_path = os.path.join(npy_dir, arch['dataset'] + '_' + arch['enc_layer'] + arch['encode_method'])
    
    input_npy_file = npy_path + '_inputs.npy'
    target_npy_file = npy_path + '_targets.npy'
    pre_enc_npy_file = npy_path + '_pre_enc.npy'
    enc_npy_file = npy_path + '_enc_data.npy'
    out_log_file = npy_path + '_out_log.npy'
    out_pre_log_file = npy_path + '_out_pre_log.npy'
    measure_data_file = npy_path + '_measure_data.npy'

    # import pdb; pdb.set_trace()
    predict_label = np.argmax(out_data, axis=1)
    acc = np.sum(np.equal(predict_label, np.array(target_data))) / len(target_data)
    # print('Acc:', acc)
    np.save(target_npy_file, np.array(target_data))
    np.save(pre_enc_npy_file, np.array(pre_data))
    if enc_data is not None:
        np.save(enc_npy_file, np.array(enc_data))
    else:
        # only empty save inputs numpy
        np.save(input_npy_file, np.array(input_data))
    np.save(out_log_file, np.array(out_data)) # after softmax
    np.save(out_pre_log_file, np.array(out_pre_data))
    np.save(measure_data_file, np.array(measure_data))

def valid_model(arch, device, model=None):
    """delete"""
    if model is None:
        model = torch.load(arch['model_file']).to(device)
        if arch['use_qiskit']:
            from torchquantum.plugins import QiskitProcessor
            processor_simulation = QiskitProcessor(use_real_qc=False)
            model.set_qiskit_processor(processor_simulation)
        if arch['use_qiskit'] and arch['use_real_qc']:
            backend_name = "ibmq_lima"
            print(f"\nTest on Real Quantum Computer {backend_name}")
            # Please specify your own hub group and project if you have the
            # IBMQ premium plan to access more machines.
            processor_real_qc = QiskitProcessor(
                use_real_qc=True,
                backend_name=backend_name,
                hub="ibm-q",
                group="open",
                project="main",
            )
            model.set_qiskit_processor(processor_real_qc)
    dataset = get_dataset(arch, arch['dataset'])
    dataflow = dict()
    for split in dataset:
        dataflow[split] = torch.utils.data.DataLoader(dataset[split], 
                                                      batch_size=512, 
                                                      shuffle=False,
                                                      num_workers=10,
                                                      pin_memory=True)
    input_data, target_data, pre_data, enc_data, out_data, out_pre_data, measure_data = [], [], [], [], [], [], []
    with torch.no_grad():
        # for split in dataset:
        #     for feed_dict in dataflow[split]:
            for feed_dict in dataflow['test']:
                if arch['data_type'] == 'nlp':
                    inputs = feed_dict[0].to(device)
                    targets = feed_dict[1].type(torch.LongTensor).to(device)
                else:
                    inputs = feed_dict[arch['data_type']].to(device)
                    targets = feed_dict['digit'].to(device)
                
                out, bsz_pre_enc, bsz_enc_data, out_pre_log = model(
                    inputs, arch)
                # print('out: ', out)
                input_data.extend(inputs.cpu().numpy())
                target_data.extend(targets.cpu().numpy())
                pre_data.extend(bsz_pre_enc.cpu().numpy())

                if bsz_enc_data is not None:
                    enc_data.extend(bsz_enc_data.cpu().numpy())
                out_data.extend(out.cpu().numpy())
                # out_pre_data.extend(out_pre_log.cpu().numpy())
                out_pre_data.extend(out_pre_log[0].cpu().numpy())
                measure_data.extend(out_pre_log[1].cpu().numpy())
    
    # save npy file
    save_npy_file(arch, input_data, target_data, pre_data, enc_data, out_data, out_pre_data, measure_data)

def valid_test_model(arch, device, model=None, dataflow=None):
    input_data, target_data, pre_data, enc_data, out_data, out_pre_data, measure_data = [], [], [], [], [], [], []
    with torch.no_grad():
        # for split in dataset:
        #     for feed_dict in dataflow[split]:
            for feed_dict in dataflow['test']:
                if arch['data_type'] == 'nlp':
                    inputs = feed_dict[0].to(device)
                    targets = feed_dict[1].type(torch.LongTensor).to(device)
                else:
                    inputs = feed_dict[arch['data_type']].to(device)
                    targets = feed_dict['digit'].to(device)
                
                out, bsz_pre_enc, bsz_enc_data, out_pre_log = model(
                    inputs, arch)
                # print('out: ', out)
                input_data.extend(inputs.cpu().numpy())
                target_data.extend(targets.cpu().numpy())
                pre_data.extend(bsz_pre_enc.cpu().numpy())

                if bsz_enc_data is not None:
                    enc_data.extend(bsz_enc_data.cpu().numpy())
                out_data.extend(out.cpu().numpy())
                # out_pre_data.extend(out_pre_log.cpu().numpy())
                out_pre_data.extend(out_pre_log[0].cpu().numpy())
                measure_data.extend(out_pre_log[1].cpu().numpy())
    
    # save npy file
    save_npy_file(arch, input_data, target_data, pre_data, enc_data, out_data, out_pre_data, measure_data)

def valid_test_model_255(arch, device, model=None, dataflow=None):
    input_data, target_data, pre_data, enc_data, out_data, out_pre_data, measure_data = [], [], [], [], [], [], []
    with torch.no_grad():
        # for split in dataset:
        #     for feed_dict in dataflow[split]:
            for feed_dict in dataflow['test']:
                if arch['data_type'] == 'nlp':
                    inputs = feed_dict[0].to(device)
                    targets = feed_dict[1].type(torch.LongTensor).to(device)
                else:
                    inputs = (feed_dict[arch['data_type']]*255).to(device)
                    targets = feed_dict['digit'].to(device)
                
                out, bsz_pre_enc, bsz_enc_data, out_pre_log = model(
                    inputs, arch)
                # print('out: ', out)
                input_data.extend(inputs.cpu().numpy())
                target_data.extend(targets.cpu().numpy())
                pre_data.extend(bsz_pre_enc.cpu().numpy())

                if bsz_enc_data is not None:
                    enc_data.extend(bsz_enc_data.cpu().numpy())
                out_data.extend(out.cpu().numpy())
                # out_pre_data.extend(out_pre_log.cpu().numpy())
                out_pre_data.extend(out_pre_log[0].cpu().numpy())
                measure_data.extend(out_pre_log[1].cpu().numpy())
    
    # save npy file
    save_npy_file(arch, input_data, target_data, pre_data, enc_data, out_data, out_pre_data, measure_data)


def reshape_data(data, datasets_num=1, layers_num=5, models_num=19):
    # convert list into numpy, reshape numpy into [encodes_num, models_num]
    data_np = np.array([np.array(data[i]) for i in range(len(data))])
    data_np = data_np.reshape(datasets_num, layers_num, models_num)
    
    return data_np

def load_total_baselines_data(data_path):
    Acc = np.squeeze(np.load(os.path.join(data_path, 'acc.npy'))) 
    MI = np.squeeze(np.load(os.path.join(data_path, 'mutual_measure.npy')))
    QSD = np.squeeze(np.load(os.path.join(data_path, 'renyi_distance_measure.npy')))
    Expre = np.squeeze(np.load(os.path.join(data_path, 'expressibility_measure.npy')))
    Entang = np.squeeze(np.load(os.path.join(data_path, 'entangling_measure.npy')))
    Ours = np.squeeze(np.load(os.path.join(data_path, 'pvi.npy')))

    # metrics, encodes, models
    data = [Acc, MI, QSD, Expre, Entang, Ours]
    data = np.array([np.array(d) for d in data]) 
    # data.shape: metrics, datasets, encodes, models

    return data

def load_single_dataset_data(data_path, dataset_name):
    Acc = np.squeeze(np.load(os.path.join(data_path, dataset_name + '_acc.npy'))) 
    MI = np.squeeze(np.load(os.path.join(data_path, dataset_name + '_mutual_measure.npy')))
    QSD = np.squeeze(np.load(os.path.join(data_path, dataset_name + '_renyi_distance_measure.npy')))
    Expre = np.squeeze(np.load(os.path.join(data_path, dataset_name + '_expressibility_measure.npy')))
    Entang = np.squeeze(np.load(os.path.join(data_path, dataset_name + '_entangling_measure.npy')))
    Ours = np.squeeze(np.load(os.path.join(data_path, dataset_name + '_pvi.npy')))

    # data.shape: metrics, encodes, models
    data = np.array([Acc, MI, QSD, Expre, Entang, Ours])

    return data

def load_single_dataset_single_model_data(data_path, dataset_name):
    Acc = np.load(os.path.join(data_path, dataset_name + '_acc.npy'))
    MI = np.load(os.path.join(data_path, dataset_name + '_mutual_measure.npy'))
    QSD = np.load(os.path.join(data_path, dataset_name + '_renyi_distance_measure.npy'))
    Expre = np.load(os.path.join(data_path, dataset_name + '_expressibility_measure.npy'))
    Entang = np.load(os.path.join(data_path, dataset_name + '_entangling_measure.npy'))
    Ours = np.load(os.path.join(data_path, dataset_name + '_pvi.npy'))

    # data.shape: metrics, encodes, models
    data = np.array([Acc, MI, QSD, Expre, Entang, Ours])

    return data

def extend_one_sort(single_dataset_matrix):
    """
        single_dataset_matrix: models, metrics, encodes
    """
    matrix = single_dataset_matrix
    for one in range(matrix.shape[0]): # models
        for two in range(1, matrix.shape[1]): # metrics
            matrix[one][two] = matrix[one][two][np.argsort(matrix[one][0])]
        matrix[one][0] = np.sort(matrix[one][0]) 
    return matrix

def save_baselines(data, result_path, dataset_name):
    """ 
        data.shape: models, metrics, encodes
    """
    baseline_res = []
    headers = ['Model ' + str(i+1) for i in range(data.shape[0])] # models
    col_names = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data.shape[2] # metrics * encodes
    models_name, metrics_name = headers, col_names
    for j in range(data.shape[0]): # models
        baseline_res.append(data[j].T.flatten())

    table = DataFrame(np.round(np.array(baseline_res), 2), \
                        columns=metrics_name, index=models_name)
    table.to_csv(os.path.join(result_path, \
                            'baseline_res_'+str(dataset_name)+'.csv') , index=True)

def load_model_data(arch):
    enc_data = np.load(os.path.join(arch['encoded_states_dir'], arch['dataset'] + '_'+\
                                    str(arch['enc_layer'])+arch['encode_method']+'_enc_data.npy'))
    measure_data = np.load(os.path.join(arch['encoded_states_dir'], arch['dataset'] + '_'+\
                                    str(arch['enc_layer'])+arch['encode_method']+'_measure_data.npy'))
    pre_enc_data = np.load(os.path.join(arch['encoded_states_dir'], arch['dataset'] + '_'+\
                                    str(arch['enc_layer'])+arch['encode_method']+'_pre_enc.npy'))
    out_log_data = np.load(os.path.join(arch['encoded_states_dir'], arch['dataset'] + '_'+\
                                    str(arch['enc_layer'])+arch['encode_method']+'_out_log.npy'))
    gt_label = np.load(os.path.join(arch['encoded_states_dir'], arch['dataset'] + '_'+\
                                    str(arch['enc_layer'])+arch['encode_method']+'_targets.npy'))
    
    return enc_data, measure_data, pre_enc_data, out_log_data, gt_label

def load_npy_file(doc_dir):
    for file in os.listdir(doc_dir):
        if '_out_log.npy' in file:
            classify_out = np.load(os.path.join(doc_dir, file))
        elif '_targets.npy' in file:
            gt_label = np.load(os.path.join(doc_dir, file))
            
    return classify_out, gt_label


def merge_data(path):
    Acc, model_mutual_info_measure, renyi_distance_info_measure, \
        expre, entangle, PVI = [], [], [], [], [], []
    datasets = ["mnist", "fashion", "cifar10", "imdb", "reuters"]
    metrics = ['amplitude', 'state_new', 'phase', 'multiphase', 'general', 'multiphase_z', 'IQP_X', 'IQP_Y', 'IQP']
    for dataset in datasets:
        # for metric in metrics:
        acc = np.load(os.path.join(path, dataset + '_acc.npy'))
        Acc.append(acc)
        model_mutual_info_measure.append(np.load(os.path.join(path, dataset + '_mutual_measure.npy')))
        renyi_distance_info_measure.append(np.load(os.path.join(path, dataset + '_renyi_distance_measure.npy')))
        expre.append(np.load(os.path.join(path, dataset + '_expressibility_measure.npy')))
        entangle.append(np.load(os.path.join(path, dataset + '_entangling_measure.npy')))
        PVI.append(np.load(os.path.join(path, dataset + '_pvi.npy')))

    Acc = np.squeeze(np.array(Acc))
    model_mutual_info_measure = np.squeeze(np.array(model_mutual_info_measure))
    renyi_distance_info_measure = np.squeeze(np.array(renyi_distance_info_measure))
    expre = np.squeeze(np.array(expre))
    entangle = np.squeeze(np.array(entangle))
    PVI = np.squeeze(np.array(PVI))

    # save metrics to npy file
    path = os.path.join(os.path.dirname(path), 'eval_whole')
    if not os.path.exists(path):
        os.makedirs(path)
    print('save path: ', path)
    np.save(os.path.join(path, 'acc.npy'), Acc)
    np.save(os.path.join(path, 'mutual_measure.npy'), model_mutual_info_measure)
    np.save(os.path.join(path, 'renyi_distance_measure.npy'), renyi_distance_info_measure)
    np.save(os.path.join(path, 'expressibility_measure.npy'), expre)
    np.save(os.path.join(path, 'entangling_measure.npy'), entangle)
    np.save(os.path.join(path, 'pvi.npy'), PVI)

    return path

def normal_data(data,min_bound=1e-8, max_bound=1.57):
    """ 
        data[bsz,n]: norm to [c, d]
    """
    # import pdb; pdb.set_trace()
    if len(data.shape) != 2:
        raise RuntimeError("data shape must be 2, but got {}".format(len(data.shape)))

    min_value, _ = torch.min(data, dim=1, keepdim=True)
    max_value, _ = torch.max(data, dim=1, keepdim=True)
    data = (data - min_value) * (max_bound - min_bound) / (max_value - min_value) + min_bound
    
    return data

def normalize_complex_data(data, min_bound=1e-8, max_bound=1.57, normal_imag=True):
    """ 
        data[bsz,n]: norm complex number to [c, d]
    """
    if len(data.shape) != 2:
        raise RuntimeError("data shape must be 2, but got {}".format(len(data.shape)))

    real_part = data.real
    imag_part = data.imag

    # Normalize the real part
    min_real, _ = torch.min(real_part, dim=1, keepdim=True)
    max_real, _ = torch.max(real_part, dim=1, keepdim=True)
    real_part = (real_part - min_real) * (max_bound - min_bound) / (max_real - min_real) + min_bound

    # Normalize the imaginary part
    if normal_imag:
        min_imag, _ = torch.min(imag_part, dim=1, keepdim=True)
        max_imag, _ = torch.max(imag_part, dim=1, keepdim=True)
        imag_part = (imag_part - min_imag) * (max_bound - min_bound) / (max_imag - min_imag) + min_bound

    # Combine the real and imaginary parts
    normalized_data = real_part + 1j * imag_part

    return normalized_data
