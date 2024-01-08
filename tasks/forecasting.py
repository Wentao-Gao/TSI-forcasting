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

   
    train_X, valid_X, test_X = load_and_preprocess_ica_data('ETTh1')

    input_dim = train_X.shape[1]
    hidden_dim = 100
    #source_dim = train_X.shape[1]
    source_dim = 50

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


    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]


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

    print("train_repr shape:", train_repr.shape)
    print("train_source_data shape before unsqueeze:", train_source_data.shape)

    #train_source_data = train_source_data.unsqueeze(0)
    print("train_source_data shape after unsqueeze:", train_source_data.shape)
    
    train_repr = torch.cat((train_repr, train_source_data.unsqueeze(0)), dim=2)
    valid_repr = torch.cat((valid_repr, valid_source_data.unsqueeze(0)), dim=2)
    test_repr = torch.cat((test_repr, test_source_data.unsqueeze(0)), dim=2)


    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    encoder_infer_time = time.time() - t

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
