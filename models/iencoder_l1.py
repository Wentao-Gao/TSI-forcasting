import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler



import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


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


def l1_penalty(var):
    """计算L1正则化损失"""
    return torch.abs(var).sum()

def train_ica_model(model, train_X, epochs=50, batch_size=32, learning_rate=0.001, alpha=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_X_2d = train_X.view(train_X.shape[0], -1)

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(train_X_2d), batch_size):
            batch_X = train_X_2d[i:i+batch_size]
            optimizer.zero_grad()
            encoded = model.encode(batch_X)
            decoded = model.decoder(encoded)
            reconstruction_loss = loss_fn(decoded, batch_X)
            independence_loss = l1_penalty(encoded)  # L1正则化促进独立性
            loss = reconstruction_loss + alpha * independence_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_X_2d)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
'''
def mutual_info_loss(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_loss = -(torch.mean(t) - torch.log(torch.mean(et)))
    return mi_loss
'''





def load_and_preprocess_ica_data(file_path):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)

    n_total = len(df)
    
    #wth
    train_slice = slice(None, int(0.6 * n_total))
    valid_slice = slice(int(0.6 * n_total), int(0.8 * n_total))
    test_slice = slice(int(0.8 * n_total), None)
    '''
    train_slice = slice(None, int(0.7 * n_total))
    valid_slice = slice(int(0.7 * n_total), int(0.9 * n_total))
    test_slice = slice(int(0.9 * n_total), None)
    '''
    '''
    #etth1/2
    train_slice = slice(None, 12 * 30 * 24)
    valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
    test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    
    
    train_slice = slice(None, 12 * 30 * 24 * 4)
    valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
    test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    '''
    train_df = df.iloc[train_slice]
    valid_df = df.iloc[valid_slice]
    test_df = df.iloc[test_slice]

    scaler_X = MinMaxScaler()

    train_X = scaler_X.fit_transform(train_df)
    valid_X = scaler_X.transform(valid_df)
    test_X = scaler_X.transform(test_df)

    train_X = torch.tensor(train_X, dtype=torch.float32)
    valid_X = torch.tensor(valid_X, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)

    return train_X, valid_X, test_X



'''
def train_ica_model(model, train_X, epochs=50, batch_size=16, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_X_2d = train_X.view(train_X.shape[0], -1)

    for epoch in range(epochs):
        # 计算总批次数，确保包含最后一个不完整的批次
        total_batches = (len(train_X_2d) + batch_size - 1) // batch_size

        for i in range(total_batches):
            # 计算批次的开始和结束索引
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_X = train_X_2d[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_X)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

def load_and_preprocess_ica_data(file_path, target):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    #df = pd.read_csv(file_path, parse_dates=True)

    n_total = len(df)

    # 定义训练集、验证集和测试集的切片

    train_slice = slice(None, int(0.6 * n_total))
    valid_slice = slice(int(0.6 * n_total), int(0.8 * n_total))
    test_slice = slice(int(0.8 * n_total), None)
    
    train_slice = slice(None, 12 * 30 * 24)
    valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
    test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
   
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

def train_ica_model(model, train_X, epochs=10, batch_size=32, learning_rate=0.001):
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

        

        import pandas as pd
        import numpy as np
        import torch
        from sklearn.preprocessing import MinMaxScaler

        def load_and_preprocess_ica_data(file_path, pred_len):
            df = pd.read_csv(file_path, index_col='date', parse_dates=True)

            # 数据预处理
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df)

            # 生成特征和标签
            def generate_pred_samples(features, pred_len):
                n = features.shape[0]
                features_trimmed = features[:-pred_len, :]
                labels = np.array([features[i:i + pred_len, :].reshape(-1) for i in range(n - pred_len)])
                return features_trimmed, labels

            features, labels = generate_pred_samples(df_scaled, pred_len)

            # 将数据切分为训练集、验证集和测试集
            n_total = len(df_scaled)
            train_slice = slice(None, int(0.6 * n_total))
            valid_slice = slice(int(0.6 * n_total), int(0.8 * n_total))
            test_slice = slice(int(0.8 * n_total), None)

            train_X = torch.tensor(features[train_slice], dtype=torch.float32)
            train_y = torch.tensor(labels[train_slice], dtype=torch.float32)
            valid_X = torch.tensor(features[valid_slice], dtype=torch.float32)
            valid_y = torch.tensor(labels[valid_slice], dtype=torch.float32)
            test_X = torch.tensor(features[test_slice], dtype=torch.float32)
            test_y = torch.tensor(labels[test_slice], dtype=torch.float32)

            return train_X, valid_X, test_X, train_y, valid_y, test_y
        '''