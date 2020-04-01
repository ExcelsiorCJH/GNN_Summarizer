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

from transformers import AlbertTokenizer, AlbertModel

from sklearn.metrics import pairwise_distances


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
        hidden = Variable(torch.zeros(1, batch_size, 32))
        cell = Variable(torch.zeros(1, batch_size, 32))
        return hidden, cell
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.6, training=True)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=True)
        x = self.conv2(x, edge_index)
        x = x.view(-1, x.size(0), self.out_dim)
        
        h_0, cell = self.init_hidden(x.size(0))  # initial h_0
        
        output, h_n = self.lstm(x, (h_0, cell))
        
        # many-to-many
        output = self.fc(output)
        
        return output


class Summarizer(nn.Module):
    
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 num_heads, 
                 num_classes=2):
        super(Summarizer, self).__init__()
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.embedder = AlbertModel.from_pretrained('albert-base-v2')
        self.gat_classifier = GATClassifier(in_dim, hidden_dim, out_dim, num_heads, num_classes)

        
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
        docs = []
        for idx in range(len(offsets) - 1):
            docs.append(word_vecs[ offsets[idx]: offsets[idx]+offsets[idx+1] ])
        
        features = [torch.mean(doc, dim=1).squeeze() for doc in docs]
        return features
    
    def build_graph(self, features_list, threshold=0.2):
        '''get edge_index for GATLayer'''
        edge_index_list = []
        for features in features_list:
            cosine_matrix = 1 - pairwise_distances(features.detach().numpy(), metric="cosine")
            adj_matrix = (cosine_matrix > threshold) * 1

            G = nx.from_numpy_matrix(adj_matrix)

            e1_list = [e1 for e1, _ in list(G.edges)]
            e2_list = [e2 for _, e2 in list(G.edges)]
            edge_index = [e1_list, e2_list]
            edge_index = torch.tensor(edge_index)
            edge_index_list.append(edge_index)

        return edge_index_list
    
    def gat_dataloader(self, features_list, edge_index_list, labels_list, batch_size):
        data_list = [
            torch_geometric.data.Data(features, edge_index, y=labels)
                for features, edge_index, labels in zip(features_list, edge_index_list, labels_list)
        ]

        gat_loader = torch_geometric.data.DataLoader(data_list, batch_size=batch_size, shuffle=False)
        return gat_loader
    

    def forward(self, 
                docs, 
                offsets, 
                labels_list, 
                threshold=0.2, 
                batch_size=32):
        
        sent_tokens = self.get_tokenize(docs)
        word_vecs = self.embedder(sent_tokens)[0]
        features_list = self.get_sentence_embedding(word_vecs, offsets)
        edge_index_list = self.build_graph(features_list, threshold)
        
        # dataloader for GATLayer
        dataloader = self.gat_dataloader(features_list, edge_index_list, labels_list, batch_size)
        
        output = self.gat_classifier(next(iter(dataloader)))
        return output