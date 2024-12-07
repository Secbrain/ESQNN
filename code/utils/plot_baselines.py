import os 
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.util import extend_one_sort

__all__ = ['load_baselines_data', 'visualize_PVI', 'base_evaluation_table', \
              'plot_multidatasets_baselines', 'plot_single_model_variance', \
              'plot_score_heatmap', 'plot_avg_score_heatmap']

def load_baselines_data(data_path):
    Acc = np.load(os.path.join(data_path, 'acc.npy'))
    MI = np.load(os.path.join(data_path, 'mutual_measure.npy'))
    QSD = np.load(os.path.join(data_path, 'renyi_distance_measure.npy'))
    Expre = np.load(os.path.join(data_path, 'expressibility_measure.npy'))
    Entang = np.load(os.path.join(data_path, 'entangling_measure.npy'))
    Ours = np.load(os.path.join(data_path, 'pvi.npy'))

    # metrics, encodes, models
    data = [Acc, MI, QSD, Expre, Entang, Ours]
    data = np.array([np.array(d) for d in data])

    return data

def base_evaluation_table(data_path, result_path, \
                          encoded_methods):
    """ plot baseline evaluation table """
    # metrics, encodes, models
    # data = [Acc, MI, QSD, Expre, Entang, Ours]
    data = load_baselines_data(data_path)
    # data = np.round(data, 2) 
    data2 = np.swapaxes(data, 0, 1) # => encodes, metrics, models

    headers = ['Model ' + str(i+1) for i in range(data2.shape[-1])] # models
    col_names = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data2.shape[1] # metrics

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # modify data format, save to csv file
    data3 = data.T # => models, metrics, encodes, 
    baseline_res = []
    for i in range(data3.shape[0]):
        baseline_res.append(data3[i].flatten())

    models_name = ['Model ' + str(i+1) for i in range(data2.shape[-1])] # models
    metrics_name = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data3.shape[1] # metrics
    table = pd.DataFrame(np.round(np.array(baseline_res), 2), columns=metrics_name, index=models_name)
    table.to_csv(os.path.join(result_path, 'baseline_res.csv') , index=True)

    return data, data3

def plot_multidatasets_baselines(data_path, result_path):
    data = load_baselines_data(data_path) # data.shape: metrics, datasets, encodes, models
    print('data.shape: ',data.shape)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # modify data format
    headers = ['Model ' + str(i+1) for i in range(data.shape[-1])] # models
    col_names = ['Acc', 'MI', 'QSD', 'Expre', 'Entang', 'Ours'] * data.shape[2] # metrics * encodes
    datasets = ["mnist", "fashion", "cifar10", "imdb", "reuters"]
    data3 = np.transpose(data, (1, 3, 0, 2)) # => datasets, models, metrics, encodes

    # save to csv file
    models_name, metrics_name = headers, col_names
    for i in range(data3.shape[0]): # datasets
        baseline_res = []
        for j in range(data3.shape[1]): # models
            baseline_res.append(data3[i][j].T.flatten())

        table = pd.DataFrame(np.round(np.array(baseline_res), 2), columns=metrics_name, index=models_name)
        table.to_csv(os.path.join(result_path, 'baseline_res_'+str(datasets[i])+'.csv') , index=True)

def plot_single_model_variance(matrix, result_path):
    """ 
        plot correlation between Accuracy and other mertics
        matrix.shape: models, metrics, encodes
    """
    orgi_matrix = matrix.copy()
    # max-min normalization
    for i in range(matrix.shape[0]): # model
        for j in range(1, matrix.shape[1]): # metrics
            matrix[i][j] = (matrix[i][j] - np.min(matrix[i][j])) / (np.max(matrix[i][j]) - np.min(matrix[i][j]))
    matrix[:, 0] = orgi_matrix[:, 0].copy() # copy origin acc

    # extend sort
    matrix = extend_one_sort(matrix)

    # Step3: plot figures
    x = matrix[:, 0] * 100
    y = matrix[:, 1:]

    fig, axs = plt.subplots(4, 5, figsize=(30, 16), dpi=600)
    fig.subplots_adjust(hspace=0.4)

    # Flatten the subplots array for easy indexing
    label = ['mutual_info', 'renyi_distance', 'exprss', 'entangle', 'PVI']
    marker = ['o', 'v', 's', 'p', 'P']
    axs = axs.flatten()
    for i in range(matrix.shape[0]): # model
        for line in range(y.shape[1]):
            if label[line] == 'PVI':
                axs[i].plot(x[i], y[i][line], marker=marker[line], label=label[line], linewidth=5)
            else:
                axs[i].plot(x[i], y[i][line], marker=marker[line], label=label[line])
            
        axs[i].set_title(f"Models Num_{str(i+1)}")
        axs[i].set_xlabel("Accuracy")
        axs[i].set_ylabel("Value")

    # savefig
    plt.savefig(os.path.join(result_path, 'models_variance.png'), dpi=800)
    plt.tight_layout()
    # Show the plot
    # plt.show()

def plot_single_point_metrics(matrix, result_path):
    """ 
        note encodes metrics
        matrix.shape: models, metrics, encodes
        normalize metrics
    """
    orig_matrix = matrix.copy()
    # max-min normalization
    for i in range(matrix.shape[0]): # model
        for j in range(1, matrix.shape[1]): # metrics
            matrix[i][j] = (matrix[i][j] - np.min(matrix[i][j])) / (np.max(matrix[i][j]) - np.min(matrix[i][j]))
    matrix[:, 0] = orig_matrix[:, 0].copy() # copy origin acc

    # extend sort
    matrix = extend_one_sort(matrix)

    x = matrix[:, 0] * 100
    y = matrix[:, 1:]

    fig, axs = plt.subplots(4, 5, figsize=(30, 16), dpi=600)
    fig.subplots_adjust(hspace=0.4)

    # Flatten the subplots array for easy indexing
    label = ['mutual_info', 'renyi_distance', 'exprss', 'entangle', 'PVI']
    metrics = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] # 9 encodes
    marker = ['o', 'v', 's', 'p', 'P']
    axs = axs.flatten()
    for i in range(matrix.shape[0]): # model
        for line in range(y.shape[1]):
            if label[line] == 'PVI':
                axs[i].plot(x[i], y[i][line], marker=marker[line], label=label[line], linewidth=5)
                for num, (a, b) in enumerate(zip(x[i], y[i][line])):
                    index = np.where(np.round(orig_matrix[i, 0]*100, 5) == np.round(a,5))[0][0]
                    axs[i].text(a, b, metrics[index], ha='left', va='bottom', fontsize=20)
            else:
                axs[i].plot(x[i], y[i][line], marker=marker[line], label=label[line])
            
        axs[i].set_title(f"Models Num_{str(i+1)}")
        axs[i].set_xlabel("Accuracy")
        axs[i].set_ylabel("Value")

    # savefig
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'models_variance.png'), dpi=800)
    
    # Show the plot
    # plt.show()

def plot_accuracy_pvi_multi(arch, data_dir_list=None, figure_path=None):
    """ plot multi directories"""
    if data_dir_list is None:
        data_dir_list = [arch['train_model_dir']]
    if figure_path is None:
        # figure_path = os.path.dirname(data_path)
        figure_path = './quantum_encoding/huatu'

    for data_dir in data_dir_list:
        pvi_dir_list = os.path.join(arch['res_root'], data_dir, '0_cal_pvi')
        # pvi_dir_list = os.path.join(data_root, data_dir, '0_cal_PVI')
        # for pvi_dir in os.listdir(pvi_dir_list):
        for num, pvi_dir in enumerate(tqdm(os.listdir(pvi_dir_list), \
                                           desc='Running: '+data_dir)):
            data_path = os.path.join(pvi_dir_list, pvi_dir, '0_pvi_cal.npy')
            figure_file = os.path.join(figure_path, data_dir, \
                                       data_dir + '_' + pvi_dir + '_figure.png')
            if not os.path.exists(os.path.dirname(figure_file)):
                os.makedirs(os.path.dirname(figure_file), exist_ok=True)

            data = np.load(data_path)
            label_fontsize, order_fontsize, width = 30, 15, 0.2

            x = np.arange(19)
            acc1, acc2, acc3, acc4 = np.zeros((19)), np.zeros((19)), np.zeros((19)), np.zeros((19))
            pvi1, pvi2, pvi3, pvi4 = np.zeros((19)), np.zeros((19)), np.zeros((19)), np.zeros((19))

            for model in range(19):
                acc1[model], acc2[model], acc3[model], acc4[model] = data[model*4+0, 0], \
                    data[model*4+1, 0], data[model*4+2, 0], data[model*4+3, 0]
                pvi1[model], pvi2[model], pvi3[model], pvi4[model] = data[model*4+0, 1], \
                    data[model*4+1, 1], data[model*4+2, 1], data[model*4+3, 1]
            
            error_attri = dict(elinewidth=1, color="r", capsize=3)
            plt.figure(figsize=(100, 20), dpi=80)

            # plot accuracy
            plt.subplot(2, 1, 1)
            plt.bar(x, acc1, width = 0.2, align='center', color='#FEE527', label='General')
            plt.bar(x+0.2, acc2, width = 0.2, align='center', color='#5BC666', label='Multiphase')
            plt.bar(x+0.4, acc3, width = 0.2, align='center', color='#248E8B', label='Phase')
            plt.bar(x+0.6, acc4, width = 0.2, align='center', color='#460056', label='State')

            # sort acc & pvi
            order1 = np.argsort(np.array([acc1, acc2, acc3, acc4]), axis=0)
        
            for num in range(19):
                order1_num = [4-list(order1[:, num]).index(0), 4-list(order1[:, num]).index(1),\
                            4-list(order1[:, num]).index(2), 4-list(order1[:, num]).index(3)]
                plt.text(x[num], acc1[num]+0.01, '%.0f' % order1_num[0], ha='center', va= 'bottom', fontsize=order_fontsize)
                plt.text(x[num]+0.2, acc2[num]+0.01, '%.0f' % order1_num[1], ha='center', va= 'bottom', fontsize=order_fontsize)
                plt.text(x[num]+0.4, acc3[num]+0.01, '%.0f' % order1_num[2], ha='center', va= 'bottom', fontsize=order_fontsize)
                plt.text(x[num]+0.6, acc4[num]+0.01, '%.0f' % order1_num[3], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.xticks(x+width*1.5,list(x+1))
            plt.tick_params(labelsize=label_fontsize)
            #plt.ylim(50, 101)
            plt.grid()
            plt.legend(fontsize=label_fontsize, loc='best')
            
            plt.title(data_dir + 'Accuracy', fontsize=label_fontsize)
            plt.ylabel('Accuracy (%)', fontsize=label_fontsize)
            
            """ plot pvi """
            plt.subplot(2, 1, 2)
            plt.bar(x, pvi1, width = 0.2, align='center', color='#FEE527', label='General')
            plt.bar(x+0.2, pvi2, width = 0.2, align='center', color='#5BC666', label='Multiphase')
            plt.bar(x+0.4, pvi3, width = 0.2, align='center', color='#248E8B', label='Phase')
            plt.bar(x+0.6, pvi4, width = 0.2, align='center', color='#460056', label='State')
            order2 = np.argsort(np.array([pvi1, pvi2, pvi3, pvi4]), axis=0)

            for num in range(19):
                order2_num = [4-list(order2[:, num]).index(0), 4-list(order2[:, num]).index(1),\
                            4-list(order2[:, num]).index(2), 4-list(order2[:, num]).index(3)]
                plt.text(x[num], pvi1[num]+0.01, '%.0f' % order2_num[0], ha='center', va= 'bottom', fontsize=order_fontsize)
                plt.text(x[num]+0.2, pvi2[num]+0.01, '%.0f' % order2_num[1], ha='center', va= 'bottom', fontsize=order_fontsize)
                plt.text(x[num]+0.4, pvi3[num]+0.01, '%.0f' % order2_num[2], ha='center', va= 'bottom', fontsize=order_fontsize)
                plt.text(x[num]+0.6, pvi4[num]+0.01, '%.0f' % order2_num[3], ha='center', va= 'bottom', fontsize=order_fontsize)

            plt.xticks(x+width*1.5,list(x+1)) # 坐标轴显示
            plt.tick_params(labelsize=label_fontsize)
            #plt.ylim(50, 101)
            plt.grid()
            plt.legend(fontsize=label_fontsize, loc='best')

            plt.title(data_dir + 'PVI info', fontsize=label_fontsize)
            plt.ylabel('PVI', fontsize=label_fontsize)
            plt.savefig(figure_file, bbox_inches='tight')


def plot_accuracy_pvi_single(data_root=None, data_dir_list=None, figure_path=None):
    if data_root is None:
        data_root = './quantum_encoding/'
    if data_dir_list is None:
        data_dir_list = ['data_run_seed_42_0325', 'data_run_seed_1111_0325', 
                    'data_run_seed_42_epoch100_0325', 'data_run_seed_1111_epoch100_0325']
    data_dir_list = ['data_run_seed_42_epoch30_encode']
    if figure_path is None:
        # figure_path = os.path.dirname(data_path)
        figure_path = './quantum_encoding/huatu'

    for data_dir in data_dir_list:
        # data_path = os.path.join(data_root, data_dir, '0_cal_PVI/0_pvi_cal.npy')
        data_path = os.path.join(data_root, data_dir, '0_cal_pvi/0_pvi_cal.npy')
        data = np.load(data_path)
        label_fontsize, order_fontsize, width = 30, 15, 0.2

        x = np.arange(19)
        acc1, acc2, acc3, acc4 = np.zeros((19)), np.zeros((19)), np.zeros((19)), np.zeros((19))
        pvi1, pvi2, pvi3, pvi4 = np.zeros((19)), np.zeros((19)), np.zeros((19)), np.zeros((19))

        for model in range(19):
            acc1[model], acc2[model], acc3[model], acc4[model] = data[model*4+0, 0], \
                data[model*4+1, 0], data[model*4+2, 0], data[model*4+3, 0]
            pvi1[model], pvi2[model], pvi3[model], pvi4[model] = data[model*4+0, 1], \
                data[model*4+1, 1], data[model*4+2, 1], data[model*4+3, 1]
        
        error_attri = dict(elinewidth=1, color="r", capsize=3) # 错误标准差
        plt.figure(figsize=(100, 20), dpi=80)

        # plot accuracy
        plt.subplot(2, 1, 1)
        plt.bar(x, acc1, width = 0.2, align='center', color='#FEE527', label='General')
        plt.bar(x+0.2, acc2, width = 0.2, align='center', color='#5BC666', label='Multiphase')
        plt.bar(x+0.4, acc3, width = 0.2, align='center', color='#248E8B', label='Phase')
        plt.bar(x+0.6, acc4, width = 0.2, align='center', color='#460056', label='State')

        # sort acc & pvi
        order1 = np.argsort(np.array([acc1, acc2, acc3, acc4]), axis=0)
    
        for num in range(19):
            order1_num = [4-list(order1[:, num]).index(0), 4-list(order1[:, num]).index(1),\
                        4-list(order1[:, num]).index(2), 4-list(order1[:, num]).index(3)]
            plt.text(x[num], acc1[num]+0.01, '%.0f' % order1_num[0], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.text(x[num]+0.2, acc2[num]+0.01, '%.0f' % order1_num[1], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.text(x[num]+0.4, acc3[num]+0.01, '%.0f' % order1_num[2], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.text(x[num]+0.6, acc4[num]+0.01, '%.0f' % order1_num[3], ha='center', va= 'bottom', fontsize=order_fontsize)
        plt.xticks(x+width*1.5,list(x+1))
        plt.tick_params(labelsize=label_fontsize)
        #plt.ylim(50, 101)
        plt.grid()
        plt.legend(fontsize=label_fontsize, loc='best')
        
        plt.title(data_dir + 'Accuracy', fontsize=label_fontsize)
        plt.ylabel('Accuracy (%)', fontsize=label_fontsize)
        
        """ plot pvi """
        plt.subplot(2, 1, 2)
        plt.bar(x, pvi1, width = 0.2, align='center', color='#FEE527', label='General')
        plt.bar(x+0.2, pvi2, width = 0.2, align='center', color='#5BC666', label='Multiphase')
        plt.bar(x+0.4, pvi3, width = 0.2, align='center', color='#248E8B', label='Phase')
        plt.bar(x+0.6, pvi4, width = 0.2, align='center', color='#460056', label='State')
        order2 = np.argsort(np.array([pvi1, pvi2, pvi3, pvi4]), axis=0)

        for num in range(19):
            order2_num = [4-list(order2[:, num]).index(0), 4-list(order2[:, num]).index(1),\
                        4-list(order2[:, num]).index(2), 4-list(order2[:, num]).index(3)]
            plt.text(x[num], pvi1[num]+0.01, '%.0f' % order2_num[0], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.text(x[num]+0.2, pvi2[num]+0.01, '%.0f' % order2_num[1], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.text(x[num]+0.4, pvi3[num]+0.01, '%.0f' % order2_num[2], ha='center', va= 'bottom', fontsize=order_fontsize)
            plt.text(x[num]+0.6, pvi4[num]+0.01, '%.0f' % order2_num[3], ha='center', va= 'bottom', fontsize=order_fontsize)

        plt.xticks(x+width*1.5,list(x+1)) # 坐标轴显示
        plt.tick_params(labelsize=label_fontsize)
        #plt.ylim(50, 101)
        plt.grid()
        plt.legend(fontsize=label_fontsize, loc='best')

        plt.title(data_dir + 'PVI info', fontsize=label_fontsize)
        plt.ylabel('PVI', fontsize=label_fontsize)
        plt.savefig(os.path.join(figure_path, data_dir + '_figure.png'), bbox_inches='tight')

