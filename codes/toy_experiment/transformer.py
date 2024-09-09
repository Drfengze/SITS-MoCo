import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

import torch.nn as nn
import torch.optim as optim

os.chdir(sys.path[0])

# Position Encoding
def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).to('cuda')
    else:
        return torch.FloatTensor(sinusoid_table)
class MultiHeadFeatureSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, num_features_to_use):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert num_features_to_use > 0, "Number of features to use must be positive"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.num_features_to_use = num_features_to_use

        self.query = nn.Linear(num_features_to_use, d_model)  # 使用指定数量的前几个维度
        self.key = nn.Linear(num_features_to_use, d_model)    # 使用指定数量的前几个维度
        self.value = nn.Linear(d_model, d_model)  # 使用所有维度

        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        # self._init_lower_triangular_weights()

    def _init_lower_triangular_weights(self):
        with torch.no_grad():
            weight = self.value.weight
            bias = self.value.bias
            assert weight.size(0) == weight.size(1), "Weight matrix must be square"
            lower_triangular_weight = torch.tril(torch.ones_like(weight))
            weight.copy_(lower_triangular_weight)
            if bias is not None:
                bias.zero_()

    def forward(self, x):
        """
        Forward pass for MultiHeadFeatureSelfAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, feature_dim)

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, feature_dim = x.size()
        assert feature_dim >= self.num_features_to_use, f"Input feature dimension must be at least {self.num_features_to_use}"

        # 使用指定数量的前几个维度计算 q 和 k
        q = self.query(x[:, :, :self.num_features_to_use])  # shape: (seq_len, batch_size, d_model)
        k = self.key(x[:, :, :self.num_features_to_use])    # shape: (seq_len, batch_size, d_model)
        v = self.value(x)                                   # shape: (seq_len, batch_size, d_model)

        # 多头分割
        q = q.view(seq_len, batch_size, self.nhead, self.d_k).transpose(0, 2)  # shape: (nhead, batch_size, seq_len, d_k)
        k = k.view(seq_len, batch_size, self.nhead, self.d_k).transpose(0, 2)  # shape: (nhead, batch_size, seq_len, d_k)
        v = v.view(seq_len, batch_size, self.nhead, self.d_k).transpose(0, 2)  # shape: (nhead, batch_size, seq_len, d_k)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # shape: (nhead, batch_size, seq_len, seq_len)
        attention = torch.softmax(scores, dim=-1)                          # shape: (nhead, batch_size, seq_len, seq_len)

        # 应用注意力
        output = torch.matmul(attention, v)  # shape: (nhead, batch_size, seq_len, d_k)
        output = output.transpose(0, 2).contiguous().view(seq_len, batch_size, self.d_model)  # shape: (seq_len, batch_size, d_model)

        # 输出投影
        output = self.out_proj(output)  # shape: (seq_len, batch_size, d_model)
        
        return output

# Transformer Encoder
class CustomEncoder(nn.Module):
    def __init__(self, input_dim=4, d_model=4, nhead=1, num_layers=1, dim_feedforward=8, max_seq_len=1000, dropout=0,num_features_to_use=2):
        super(CustomEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(get_sinusoid_encoding_table(max_seq_len, input_dim))
        self.mha = nn.MultiheadAttention(d_model, nhead)
        self.encoder_layers1 = nn.ModuleList([
            MultiHeadFeatureSelfAttention(d_model, nhead,num_features_to_use=num_features_to_use)
        ])
        self.encoder_layers2 = nn.ModuleList([
            MultiHeadFeatureSelfAttention(d_model, nhead,num_features_to_use=d_model)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, src):
        src = src.transpose(0, 1)  # Change to (seq_len, batch_size, input_dim)
        embedded = self.embedding(src)
        seq_len = src.size(0)
        src2 = src + self.pos_embedding[:seq_len, :].unsqueeze(1)
        for layer in self.encoder_layers1:
            src2 = layer(src2)        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈神经网络子层
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # for layer in self.encoder_layers2:
        #     src2 = layer(src2)
        src2 = torch.cumsum(src2, dim=0)  
        src = src + self.dropout2(src2)        
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src)
        for layer in self.encoder_layers2:
            src2 = layer(src)
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src) 
        return src  #(seq_len, batch_size, input_dim)
    
class EncoderDecoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_len,num_classes=10):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = CustomEncoder(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        # self.decoder = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0),
        #     num_decoder_layers
        # )
        decoder_structure = [d_model, 64, 32, num_classes]
        layers = []
        for i in range(len(decoder_structure) - 1):
            layers.append(nn.Linear(decoder_structure[i], decoder_structure[i + 1]))
            if i < (len(decoder_structure) - 2):
                layers.extend([
                    nn.BatchNorm1d(decoder_structure[i + 1]),
                    nn.ReLU()
                ])
        self.decoder = nn.Sequential(*layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src):
        encoder_output = self.encoder(src)
        # encoder_output = torch.cumsum(encoder_output, dim=0)
        # decoder_output = encoder_output#self.decoder(encoder_output, encoder_output)
        pooled = self.global_pool(encoder_output.permute(1, 2, 0)).squeeze(-1) # 全局平均池化, shape: (batch_size, d_model)
        return self.decoder(pooled)
    
# Model Training
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.014):
    # use apple silicon GPU if available    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model