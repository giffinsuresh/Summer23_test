# Your models
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MyNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = out.view(out.shape[0], 15, 5)
        return out
