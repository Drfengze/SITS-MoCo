import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)

class CopeAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.cope = CoPE(npos_max=1000, head_dim=self.head_dim)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        seq_len, batch_size, _ = x.size()
        q = self.query(x).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 2)
        k = self.key(x).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 2)
        v = self.value(x).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_logits += self.cope(q, attn_logits)
        attn = torch.softmax(attn_logits, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(0, 2).contiguous().view(seq_len, batch_size, self.d_model)
        return self.out_proj(output)

class CustomEncoder(nn.Module):
    def __init__(self, input_dim=4, d_model=4, nhead=1, num_layers=1, dim_feedforward=8, dropout=0):
        super(CustomEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layers = nn.ModuleList([
            CopeAttention(d_model, nhead)
            for _ in range(num_layers)
        ])
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
        src = self.embedding(src)
        for layer in self.encoder_layers:
            src2 = layer(src)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

class EncoderDecoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, num_classes=10):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = CustomEncoder(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
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

    def forward(self, src):
        encoder_output = self.encoder(src)
        pooled = self.global_pool(encoder_output.permute(1, 2, 0)).squeeze(-1)
        return self.decoder(pooled)

# Model Training function remains the same
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.014):
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