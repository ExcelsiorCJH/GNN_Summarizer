import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class LSTM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_dim=256,
                 hidden_dim=128,
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.2):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if bidirectional:
            self.num_directs = 2
        else:
            self.num_directs = 1
        
        self.dropout = dropout
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, 
                              num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(self.num_directs*hidden_dim, hidden_dim)
        
    
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(
            torch.zeros(self.num_layers*self.num_directs, batch_size, self.hidden_dim)
        )
        
        cell = Variable(
            torch.zeros(self.num_layers*self.num_directs, batch_size, self.hidden_dim)
        )
        return hidden, cell
        

    def forward(self, sents):
        x = self.embed(sents)
        
        h_0, cell = self.init_hidden(x.size(0))  # initial h_0
        h_0, cell = h_0.to(DEVICE), cell.to(DEVICE)

        # (batch, seq, feature)
        output, h_n = self.bilstm(x, (h_0, cell))
        output = torch.mean(output, dim=1)
        output = self.linear(output)
        return output