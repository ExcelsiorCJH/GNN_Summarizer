import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import torch_geometric
from torch_geometric.nn import GATConv


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_classes=1):
        super().__init__()
        
        self.out_head = 1
        self.out_dim = out_dim
        
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, concat=False,
                             heads=self.out_head, dropout=0.6)
        
        self.lstm = nn.LSTM(out_dim, 32, 1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(32, num_classes)
        
    
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(1, batch_size, 32), )
        cell = Variable(torch.zeros(1, batch_size, 32))
        return hidden, cell
    

    def forward(self, data):
        x, edge_index = data.x.to(DEVICE), data.edge_index.to(DEVICE)
        
        x = F.dropout(x, p=0.6, training=True)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=True)
        x = self.conv2(x, edge_index)
        x = x.view(-1, x.size(0), self.out_dim)
        
        h_0, cell = self.init_hidden(x.size(0))  # initial h_0
        h_0, cell = h_0.to(DEVICE), cell.to(DEVICE)
        output, h_n = self.lstm(x, (h_0, cell))
        
        # many-to-many
        output = self.fc(output)
        
        return output