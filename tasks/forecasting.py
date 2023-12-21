import numpy as np
import time
from . import _eval_protocols as eval_protocols
from models.iencoder import load_and_preprocess_ica_data, NonlinearICA, train_ica_model

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F



# train_repr, train_data, pred_len, drop=padding\
# 得到表征， 以及标签（y的真实值），也就是表征预测标签 这里要加上ICA的表征
def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])


def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }

def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, padding):

    #file_path = r'C:\Users\gwt99\Downloads\CoST-main\CoST-main\datasets\WTH.csv'
    #file_path = r'C:\Users\gwt99\Downloads\CoST-main\CoST-main\tcl_output'
    #C:\Users\gwt99\PycharmProjects\TSI(Trend Seasonality Indenpendent)\datasets\ETTm1.csv
    file_path = r'C:\Users\gwt99\PycharmProjects\TSI(Trend Seasonality Indenpendent)\datasets\ETTm2.csv'

    #file_path = r'C:\Users\gwt99\PycharmProjects\TSI(Trend Seasonality Indenpendent)\datasets\electricity.csv'
    #C:\Users\gwt99\PycharmProjects\TSI(Trend Seasonality Indenpendent)\datasets\exchange_rate.csv

    #target = 'RelativeHumidity'
    ## Visibility  HUFL
############################################################################################################
    #target = 'WetBulbCelsius'
    #target = 'OT'Visibility WetBulbCelsius
    #df = pd.read_csv(file_path)
    #print('target shape', df.shape)

    #train_X, valid_X, test_X, train_y, valid_y, test_y = load_and_preprocess_ica_data(file_path, target)
    #train_X, valid_X, test_X, train_y, valid_y, test_y = load_and_preprocess_ica_data(file_path, 24)
    train_X, valid_X, test_X = load_and_preprocess_ica_data('ETTh2')

    input_dim = train_X.shape[1]
    hidden_dim = 100
    #source_dim = train_X.shape[1]
    source_dim = 50
    '''
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        elif name == 'WTH':
            data = data[['WetBulbCelsius']]
        else:
            data = data.iloc[:, -1:]
    '''

    ica_model = NonlinearICA(input_dim, hidden_dim, source_dim)
    train_ica_model(ica_model, train_X)

    train_source_data = ica_model.encode(train_X.view(train_X.shape[0], -1)).detach().numpy()
    valid_source_data = ica_model.encode(valid_X.view(valid_X.shape[0], -1)).detach().numpy()
    test_source_data = ica_model.encode(test_X.view(test_X.shape[0], -1)).detach().numpy()
############################################################################################################

    t = time.time()

    # Trend and Seasonality
    all_repr = model.encode(
        data,
        mode='forecasting',
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )

    # 把表征分成训练集，验证集与测试集  这里要加上ICA的表征
    # 分割表征为训练集，验证集与测试集
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    # 确保所有的数据都是 PyTorch 张量
    train_repr = torch.from_numpy(train_repr).float() if isinstance(train_repr, np.ndarray) else train_repr
    valid_repr = torch.from_numpy(valid_repr).float() if isinstance(valid_repr, np.ndarray) else valid_repr
    test_repr = torch.from_numpy(test_repr).float() if isinstance(test_repr, np.ndarray) else test_repr

    train_source_data = torch.from_numpy(train_source_data).float() if isinstance(train_source_data,
                                                                                  np.ndarray) else train_source_data
    valid_source_data = torch.from_numpy(valid_source_data).float() if isinstance(valid_source_data,
                                                                                  np.ndarray) else valid_source_data
    test_source_data = torch.from_numpy(test_source_data).float() if isinstance(test_source_data,
                                                                                np.ndarray) else test_source_data
    ##################################################
    '''
    #只有electricity需要
    # 转置 train_source_data
    train_source_data = train_source_data.T  # 现在形状变为 [321, 15782]

    # 在第2维进行 unsqueeze
    train_source_data = train_source_data.unsqueeze(2)  # 现在形状变为 [321, 15782, 1]

    # 验证集数据处理
    valid_source_data = valid_source_data.T  # 转置
    valid_source_data = valid_source_data.unsqueeze(2)  # 增加维度


    # 测试集数据处理
    test_source_data = test_source_data.T  # 转置
    test_source_data = test_source_data.unsqueeze(2)  # 增加维度
    '''

    # 打印维度以便于调试
    print("train_repr shape:", train_repr.shape)
    print("train_source_data shape before unsqueeze:", train_source_data.shape)

    # 调整维度
    #train_source_data = train_source_data.unsqueeze(0)
    print("train_source_data shape after unsqueeze:", train_source_data.shape)

    # 如果需要，可以进一步调整 train_repr 或 train_source_data 的维度
    # 例如：train_repr = train_repr.reshape(-1, 1, train_repr.shape[-1])

    ##这个是对的
    # 现在可以使用 torch.cat 来合并张量, 现在表征就包含了ICA，普通ica
    train_repr = torch.cat((train_repr, train_source_data.unsqueeze(0)), dim=2)
    valid_repr = torch.cat((valid_repr, valid_source_data.unsqueeze(0)), dim=2)
    test_repr = torch.cat((test_repr, test_source_data.unsqueeze(0)), dim=2)

    ##################################################

    '''
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    # 假设 X 是您的高维特征矩阵，每行是一个数据点的表征
    X = train_repr  # 替换为您的表征数据

    # Reshape X to a 2D array
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Apply StandardScaler to the 2D array
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape the scaled data back to the original shape
    X_scaled_reshaped = X_scaled.reshape(X.shape)


    # 应用 t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X_scaled)

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='.', alpha=0.7)
    plt.title('t-SNE visualization of representations')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()
    '''


    ##################################################

    #Electricity
    #train_repr = torch.cat((train_repr, train_source_data.unsqueeze(2)), dim=2)
    #valid_repr = torch.cat((valid_repr, valid_source_data.unsqueeze(2)), dim=2)
    #test_repr = torch.cat((test_repr, test_source_data.unsqueeze(2)), dim=2)

    print("train_source_data shape after unsqueeze:", train_source_data.shape)
    #train_repr = train_source_data.unsqueeze(0)
    #valid_repr = valid_source_data.unsqueeze(0)
    #test_repr = test_source_data.unsqueeze(0)


    # 与表征对应的数据
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    encoder_infer_time = time.time() - t

    # 记录结果
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        #lr = eval_protocols.fit_kernel_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'encoder_infer_time': encoder_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
