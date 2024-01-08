import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler



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


def load_and_preprocess_ica_data(name, univar=False):
    #df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)

    n_total = len(df)

    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate'):
            df = df[['OT']]
        elif name == 'electricity':
            df = df[['MT_001']]
        elif name == 'WTH':
            df = df[['WetBulbCelsius']]
        elif name == 'weather_case':
            df = df[['prate']]
        else:
            df = df.iloc[:, -1:]

    #df = df.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    elif name.startswith('M5'):
        train_slice = slice(None, int(0.8 * (1913 + 28)))
        valid_slice = slice(int(0.8 * (1913 + 28)), 1913 + 28)
        test_slice = slice(1913 + 28 - 1, 1913 + 2 * 28)
    else:
        train_slice = slice(None, int(0.6 * len(df)))
        valid_slice = slice(int(0.6 * len(df)), int(0.8 * len(df)))
        test_slice = slice(int(0.8 * len(df)), None)


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

def train_ica_model(model, train_X, epochs=20, batch_size=16, learning_rate=0.001):
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


