import pandas as pd
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
            independence_loss = l1_penalty(encoded)  
            loss = reconstruction_loss + alpha * independence_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_X_2d)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")



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


