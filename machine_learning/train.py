# Write Data loaders, training procedure and validation procedure in this file.

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

train_x = pd.read_csv('train_x.csv',header=None, index_col=False)
val_x = pd.read_csv('val_x.csv',header=None, index_col=False)
train_y = pd.read_csv('train_y.csv',header=None, index_col=False)
val_y = pd.read_csv('val_y.csv',header=None, index_col=False)

train_x = train_x.values
val_x = val_x.values
train_y = train_y.values
val_y = val_y.values

def format_input(x):
    new_x = []
    for row in range(x.shape[0]):
        temp = []
        for col in range(x.shape[1]):
            temp.append(ast.literal_eval(x[row][col])) 
        new_x.append(temp)
    return np.array(new_x)
    
train_x = format_input(train_x)
val_x = format_input(val_x)
train_y = format_input(train_y)
val_y = format_input(val_y)

print(train_x.shape)

train_x_tensor = torch.Tensor(train_x)
val_x_tensor = torch.Tensor(val_x)
train_y_tensor = torch.Tensor(train_y)
val_y_tensor = torch.Tensor(val_y)

print(train_x_tensor.shape)
print(val_x_tensor.shape)
print(train_y_tensor.shape)
print(val_y_tensor.shape)

train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM model
input_dim = 5
hidden_dim = 64
output_dim = 5
num_layers = 2

model = MyNet(input_dim, hidden_dim, output_dim, num_layers).to(device)

# Define hyperparameters
lr = 0.01
batch_size = 64
num_epochs = 5

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train model
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
    
    # Print results
    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss))

torch.save(model.state_dict(), "model_weights.pth")



