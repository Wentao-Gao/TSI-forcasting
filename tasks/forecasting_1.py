import numpy as np
import time
from . import _eval_protocols as eval_protocols

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


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

class NonlinearICA(nn.Module):
    def __init__(self, input_dim, hidden_dim, source_dim):
        super(NonlinearICA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, source_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

def load_and_preprocess_data(file_path, target):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    #df = pd.read_csv(file_path, parse_dates=True)

    n_total = len(df)

    # 定义训练集、验证集和测试集的切片

    train_slice = slice(None, int(0.6 * n_total))
    valid_slice = slice(int(0.6 * n_total), int(0.8 * n_total))
    test_slice = slice(int(0.8 * n_total), None)
    '''
    train_slice = slice(None, 12 * 30 * 24)
    valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
    test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    '''
    # 切分数据
    train_df = df.iloc[train_slice]
    valid_df = df.iloc[valid_slice]
    test_df = df.iloc[test_slice]

    # 数据预处理
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_X = scaler_X.fit_transform(train_df.drop(target, axis=1))
    train_y = scaler_y.fit_transform(train_df[[target]])

    valid_X = scaler_X.transform(valid_df.drop(target, axis=1))
    valid_y = scaler_y.transform(valid_df[[target]])

    test_X = scaler_X.transform(test_df.drop(target, axis=1))
    test_y = scaler_y.transform(test_df[[target]])

    # 转换为 PyTorch 张量
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    valid_X = torch.tensor(valid_X, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    return train_X, valid_X, test_X, train_y, valid_y, test_y

def train_model(model, train_X, epochs=100, batch_size=32, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_X_2d = train_X.view(train_X.shape[0], -1)

    for epoch in range(epochs):
        for i in range(0, len(train_X_2d), batch_size):
            batch_X = train_X_2d[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_X)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, padding):

    file_path = r'C:\Users\gwt99\Downloads\CoST-main\CoST-main\datasets\WTH.csv'
    #file_path = r'C:\Users\gwt99\Downloads\CoST-main\CoST-main\tcl_output'

    #file_path = r'C:\Users\gwt99\Downloads\CoST-main\CoST-main\datasets\ETTh2.csv'
    #target = 'RelativeHumidity'
    ## Visibility  HUFL
    target = 'Visibility'
    #target = 'OT'Visibility
    df = pd.read_csv(file_path)
    print('target shape', df.shape)

    # 加载 df2 数据
    #df2 = pd.read_csv(r'C:\Users\gwt99\Downloads\CoST-main\CoST-main\tcl_output')

    # 预处理 df2（例如，规范化、填充缺失值等）
    # 这里需要根据您的具体数据和需求来定制
    # df2 = preprocess(df2) # 假设 preprocess 是一个自定义函数

    # 计算每个部分的索引
    #train_end_idx = int(0.6 * len(df2))
    #valid_end_idx = int(0.8 * len(df2))

    # 使用切片来划分 df2
    #df2_train = df2.iloc[:train_end_idx]
    #df2_valid = df2.iloc[train_end_idx:valid_end_idx]
    #df2_test = df2.iloc[valid_end_idx:]

    # 将这些划分转换为 PyTorch 张量
    #df2_train_tensor = torch.tensor(df2_train.values).float()
    #df2_valid_tensor = torch.tensor(df2_valid.values).float()
    #df2_test_tensor = torch.tensor(df2_test.values).float()

    # 调整张量的形状以匹配 train_repr, valid_repr, test_repr
    # 这取决于您的具体情况
    # 例如，如果需要，可以使用 view() 或 reshape()


    # 确保维度匹配
    # 这一步可能需要根据您的具体情况调整
    # 例如，如果 df2_tensor 需要被加到每个时间步，您可能需要调整其形状以匹配 train_repr 的形状
    #df2_tensor = df2_tensor.view(train_repr.shape[0], -1, df2_tensor.shape[-1])

    # 合并 df2_tensor 与 train_repr
    #train_repr = torch.cat((train_repr, df2_tensor), dim=2)

    train_X, valid_X, test_X, train_y, valid_y, test_y = load_and_preprocess_data(file_path, target)

    input_dim = train_X.shape[1]
    hidden_dim = 10
    source_dim = 10

    ica_model = NonlinearICA(input_dim, hidden_dim, source_dim)
    train_model(ica_model, train_X)

    train_source_data = ica_model.encode(train_X.view(train_X.shape[0], -1)).detach().numpy()
    valid_source_data = ica_model.encode(valid_X.view(valid_X.shape[0], -1)).detach().numpy()
    test_source_data = ica_model.encode(test_X.view(test_X.shape[0], -1)).detach().numpy()

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

    # 打印维度以便于调试
    print("train_repr shape:", train_repr.shape)
    print("train_source_data shape before unsqueeze:", train_source_data.shape)

    # 调整维度
    #train_source_data = train_source_data.unsqueeze(0)
    print("train_source_data shape after unsqueeze:", train_source_data.shape)

    # 如果需要，可以进一步调整 train_repr 或 train_source_data 的维度
    # 例如：train_repr = train_repr.reshape(-1, 1, train_repr.shape[-1])

    '''
    # 合并 df2_tensor 与 train_repr
    # 合并张量
    train_repr = torch.cat((train_repr, df2_train_tensor.unsqueeze(0)), dim=2)
    valid_repr = torch.cat((valid_repr, df2_valid_tensor.unsqueeze(0)), dim=2)
    test_repr = torch.cat((test_repr, df2_test_tensor.unsqueeze(0)), dim=2)
    '''



    # 现在可以使用 torch.cat 来合并张量, 现在表征就包含了ICA，普通ica
    train_repr = torch.cat((train_repr, train_source_data.unsqueeze(0)), dim=2)
    valid_repr = torch.cat((valid_repr, valid_source_data.unsqueeze(0)), dim=2)
    test_repr = torch.cat((test_repr, test_source_data.unsqueeze(0)), dim=2)

    print("train_source_data shape after unsqueeze:", train_source_data.shape)

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
