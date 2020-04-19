import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import torch_geometric
from torch_geometric.nn import GATConv

from transformers import AlbertTokenizer, AlbertModel, AlbertConfig

from sklearn.metrics import pairwise_distances

from lstm import LSTM
from gat import GATClassifier


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")



class NaiveSummarizer(nn.Module):
    
    def __init__(self,  
                 num_classes=1):
        super(NaiveSummarizer, self).__init__()
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.tokenizer.padding_side = 'left'
        
        self.embedder = LSTM(self.tokenizer.vocab_size)
        self.lstm = nn.LSTM(128, 64, 1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(64, num_classes)
        

        
    def get_tokenize(self, docs):
        sent_tokens = [
            torch.cat(
                [self.tokenizer.encode(
                        sentences[i],
                        add_special_tokens=True,
                        max_length=64,
                        pad_to_max_length=True,
                        return_tensors='pt'
                 ) for i in range(len(sentences))]
            ) for sentences in docs
        ]

        sent_tokens = torch.cat([*sent_tokens])
        return sent_tokens
    
    def get_sentence_embedding(self, word_vecs, offsets):
        '''get node-featrues(setences embedding)'''
        features = []
        for idx in range(len(offsets) - 1):
            features.append(word_vecs[ offsets[idx]: offsets[idx]+offsets[idx+1] ])
        
        maxlen = max(offsets)
        features = [feature.cpu().detach().numpy() for feature in features]
        
        pad_features = []
        for feature in features:
            pad_len = maxlen - len(feature)
            pad_features.append(
                np.concatenate((np.zeros((pad_len, 128)), feature), axis=0)
            )
            
        
        return torch.tensor(pad_features, dtype=torch.float32).to(DEVICE)
    
    
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(1, batch_size, 64))
        cell = Variable(torch.zeros(1, batch_size, 64))
        return hidden, cell
    

    def forward(self, 
                docs, 
                offsets, 
                labels_list, 
                threshold=0.2, 
                batch_size=32):
        
        sent_tokens = self.get_tokenize(docs).to(DEVICE)
        word_vecs = self.embedder(sent_tokens)
        features = self.get_sentence_embedding(word_vecs, offsets)
        
        h_0, cell = self.init_hidden(features.size(0))  # initial h_0
        h_0, cell = h_0.to(DEVICE), cell.to(DEVICE)
        
        output, h_n = self.lstm(features, (h_0, cell))
        output = self.fc(output)
        return output