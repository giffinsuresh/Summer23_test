#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MyNet

test_x = pd.read_csv('test_x.csv',header=None, index_col=False)
test_y = pd.read_csv('test_y.csv',header=None, index_col=False)

test_x = test_x.values
test_y = test_y.values

def format_input(x):
    new_x = []
    for row in range(x.shape[0]):
        temp = []
        for col in range(x.shape[1]):
            temp.append(ast.literal_eval(x[row][col])) 
        new_x.append(temp)
    return np.array(new_x)
    
test_x = format_input(test_x)
test_y = format_input(test_y)

print(test_x.shape)

test_x_tensor = torch.Tensor(test_x)
test_y_tensor = torch.Tensor(test_y)

print(test_x_tensor.shape)
print(test_y_tensor.shape)

test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 5
hidden_dim = 64
output_dim = 5
num_layers = 2
model = MyNet(input_dim, hidden_dim, output_dim, num_layers).to(device)
model.load_state_dict(torch.load("model_weights.pth"))

# Evaluate on Test
model.eval()
criterion = nn.MSELoss()
test_loss = 0.0
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)

# Print results
print("Test Loss",test_loss)







