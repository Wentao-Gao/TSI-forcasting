import torch
from torch import nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_heads, dim_feedforward, max_seq_length, dropout=0.1,
                 extract_layers=None):
        super().__init__()

        self.extract_layers = extract_layers
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_length, in_channels))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        # 添加位置编码
        seq_length = x.size(2)
        x = x + self.positional_encoding[:seq_length, :].unsqueeze(0)

        # 转换维度以符合Transformer的期望输入 (batch, seq, feature)
        x = x.transpose(1, 2)  # 从 (batch, feature, seq) 转换到 (batch, seq, feature)

        # 如果需要，提取特定层的输出
        if self.extract_layers is not None:
            outputs = []
            for idx in range(self.transformer_encoder.num_layers):
                x = self.transformer_encoder.layers[idx](x)
                if idx in self.extract_layers:
                    outputs.append(x.transpose(1, 2))  # 转换回 (batch, feature, seq)
            return outputs

        x = self.transformer_encoder(x)
        return x.transpose(1, 2)  # 转换回 (batch, feature, seq)

