import torch
from torch import nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, bidirectional=False, extract_layers=None):
        super().__init__()

        self.extract_layers = extract_layers
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=10,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.out_channels = hidden_channels * 2 if bidirectional else hidden_channels

        if extract_layers is not None:
            assert max(extract_layers) < num_layers

    def forward(self, x):
        # LSTM expects input of shape (batch, seq, feature)
        x = x.transpose(1, 2)  # Convert from (batch, feature, seq) to (batch, seq, feature)

        if self.extract_layers is not None:
            outputs = []
            for layer in range(self.lstm.num_layers):
                x, _ = self.lstm(x)
                if layer in self.extract_layers:
                    outputs.append(x.transpose(1, 2))  # Convert back to (batch, feature, seq)
            return outputs

        x, _ = self.lstm(x)
        return x.transpose(1, 2)  # Convert back to (batch, feature, seq)



class BiLSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, extract_layers=None):
        super().__init__()

        self.extract_layers = extract_layers
        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.out_channels = hidden_channels * 2  # 由于是双向，所以输出通道数是隐藏通道数的两倍

        if extract_layers is not None:
            assert max(extract_layers) < num_layers

    def forward(self, x):
        # LSTM期望的输入格式为(batch, seq, feature)
        x = x.transpose(1, 2)  # 将输入从(batch, feature, seq)转换为(batch, seq, feature)

        all_outputs = []
        for layer in range(self.bilstm.num_layers):
            x, _ = self.bilstm(x)
            all_outputs.append(x)

        if self.extract_layers is not None:
            extracted_outputs = [all_outputs[i].transpose(1, 2) for i in self.extract_layers]  # 转换回(batch, feature, seq)格式
            return extracted_outputs

        return all_outputs[-1].transpose(1, 2)  # 只返回最后一层的输出，并转换回(batch, feature, seq)
