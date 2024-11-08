# Data Simulation
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
os.chdir(sys.path[0])
def improved_matrix_independent_accumulation(signal1, signal2, threshold=0.1):
    n = len(signal1)
    
    # Step 1: Create a mask for where signal1 is above the threshold
    mask = (signal1 > threshold).astype('float32')
    
    # Step 2: Detect the start of each peak
    peak_starts = np.diff(mask, prepend=0) > 0
    
    # Step 3: Create a cumulative sum of signal2
    cumsum_signal2 = np.cumsum(signal2 * mask)
    
    # Step 4: Create a matrix to subtract the cumsum at the start of each peak
    subtract_matrix = np.tril(np.ones((n, n)))
    subtract_values = cumsum_signal2 * peak_starts
    subtract_matrix = subtract_matrix * subtract_values
    
    # Step 5: Compute the final result
    result = cumsum_signal2 - np.sum(subtract_matrix, axis=1)
    
    # Step 6: Apply the mask to keep only values during peaks
    result *= mask
    
    return result

def generate_multi_peak_signal(length, num_peaks):
    signal = np.zeros(length)
    for _ in range(num_peaks):
        peak_loc = np.random.randint(0, length)
        peak_width = np.random.randint(10, 50)
        peak_height = np.random.uniform(0.5, 1.0)
        x = np.arange(length)
        signal += peak_height * np.exp(-0.5 * ((x - peak_loc) / peak_width) ** 2)
    return signal


n=365 
signal1_lst = []
signal2_lst = []
ground_truth_lst = []
batch_size = 30000
for i in range(1, batch_size):
    signal1 = generate_multi_peak_signal(n, 2)
    signal2 = np.random.randn(n)
    ground_truth = improved_matrix_independent_accumulation(signal1, signal2).reshape(-1, 1)
    cum_signal1 = signal1.reshape(-1, 1)
    signal1 = np.concatenate([signal1.reshape(-1, 1), cum_signal1], axis=1)
    cum_signal2 = signal2.reshape(-1, 1)
    signal2 = np.concatenate([signal2.reshape(-1, 1), cum_signal2], axis=1)
    signal1_lst.append(signal1)
    signal2_lst.append(signal2)
    ground_truth_lst.append(ground_truth)

signal1 = torch.tensor(np.stack(signal1_lst).astype('float32'))
signal2 = torch.tensor(np.stack(signal2_lst).astype('float32'))
ground_truth = torch.tensor(np.stack(ground_truth_lst).astype('float32'))

# Ensure signals are 3D
if signal1.dim() == 2:
    signal1 = signal1.unsqueeze(-1)
if signal2.dim() == 2:
    signal2 = signal2.unsqueeze(-1)

# 数据集分割
train_size = int(0.7 * batch_size)
val_size = int(0.2 * batch_size)
test_size = batch_size - train_size - val_size

train_data = TensorDataset(
    torch.cat([signal1[:train_size], signal2[:train_size]], dim=2),
    ground_truth[:train_size]
)
val_data = TensorDataset(
    torch.cat([signal1[train_size:train_size+val_size], signal2[train_size:train_size+val_size]], dim=2),
    ground_truth[train_size:train_size+val_size]
)
test_data = TensorDataset(
    torch.cat([signal1[train_size+val_size:], signal2[train_size+val_size:]], dim=2),
    ground_truth[train_size+val_size:]
)

# save the data
torch.save(train_data, 'toy_data/train_data.pth')
torch.save(val_data, 'toy_data/val_data.pth')
torch.save(test_data, 'toy_data/test_data.pth')