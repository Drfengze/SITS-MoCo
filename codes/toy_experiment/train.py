import wandb
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformer import *  
import matplotlib.pyplot as plt

# 初始化 wandb
wandb.init(project="transformer",name="cope_transformer")
# 数据生成和准备 (使用你原来的代码)
n = 365
# wandb 配置
config = wandb.config
config.learning_rate = 0.008
config.epochs = 200
config.batch_size = 360
config.nhead = 1
config.d_model = 4
config.num_encoder_layers = 1
config.dim_feedforward = 8
# load the data
train_data = torch.load('toy_data/train_data.pth')
val_data = torch.load('toy_data/val_data.pth')

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.batch_size)

# 模型定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoderModel(input_dim=4, d_model=config.d_model, nhead=config.nhead, num_encoder_layers=config.num_encoder_layers, dim_feedforward=config.dim_feedforward, max_seq_len=1000, num_decoder_layers=1)
model.to(device)

# 优化器和损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs):
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
        
        # 记录到 wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader)
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

# 开始训练
trained_model = train_model(model, train_loader, val_loader, num_epochs=config.epochs)

# 保存模型
torch.save(trained_model.state_dict(), "trained_model.pth")
wandb.save("trained_model.pth")

test_data = torch.load('toy_data/test_data.pth')
# 测试模型
test_input, test_ground_truth = test_data.tensors

with torch.no_grad():
    model_output = trained_model(test_input.to(device))


# 可视化结果
plt.figure(figsize=(5, 10))
#获取随机整数n
n = np.random.randint(0, config.batch_size )
plt.subplot(3, 1, 1)
plt.title("Smooth Multi-peak Signal (Signal 1)")
plt.plot(test_input[n, :, 0].numpy())

plt.subplot(3, 1, 2)
plt.title("Random Signal (Signal 2)")
plt.plot(test_input[n, :, 2].numpy())

plt.subplot(3, 1, 3)
plt.title("Ground Truth")
plt.plot(test_ground_truth[n, :, 0].numpy())

plt.plot(model_output[n, :, 0].cpu().numpy().transpose())
plt.tight_layout()

wandb.log({"results": wandb.Image(plt)})

# 结束 wandb 运行
wandb.finish()