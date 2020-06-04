import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix
from torch_geometric.utils import to_networkx

from transformers import AlbertTokenizer, AlbertModel, AlbertConfig

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


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


class BasicSummarizer(nn.Module):
    
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 num_heads, 
                 num_classes=2):
        super(BasicSummarizer, self).__init__()
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.tokenizer.padding_side = 'left'
        self.embedder = LSTM(self.tokenizer.vocab_size)
        self.gat_classifier = GATClassifier(in_dim, hidden_dim, out_dim, num_heads, num_classes)

        
    def get_tokenize(self, docs):
        sent_tokens = [
            torch.cat(
                [self.tokenizer.encode(
                        sentences[i],
                        add_special_tokens=True,
                        max_length=128,
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
        
        return features
    
    def build_graph(self, features_list, threshold=0.2):
        '''get edge_index for GATLayer'''
        edge_index_list = []
        for features in features_list:
            features = features.cpu()
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
        
        sent_tokens = self.get_tokenize(docs).to(DEVICE)
        word_vecs = self.embedder(sent_tokens)
        features_list = self.get_sentence_embedding(word_vecs, offsets)
        edge_index_list = self.build_graph(features_list, threshold)
        
        # dataloader for GATLayer
        dataloader = self.gat_dataloader(features_list, edge_index_list, labels_list, batch_size)
        
        output = self.gat_classifier(next(iter(dataloader)))
        return output

############################  BasicSummarizerWithGDC   ############################

class BasicSummarizerWithGDC(nn.Module):
    
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 num_heads, 
                 num_classes=2,
                 use_gdc=True):
        super(BasicSummarizerWithGDC, self).__init__()
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.tokenizer.padding_side = 'left'
        self.embedder = LSTM(self.tokenizer.vocab_size)
        self.gat_classifier = GATClassifier(in_dim, hidden_dim, out_dim, num_heads, num_classes)
        self.use_gdc = use_gdc
        if self.use_gdc:
            self.gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                            normalization_out='col',
                            diffusion_kwargs=dict(method='ppr', alpha=0.05),
                            sparsification_kwargs=dict(method='topk', k=5,
                                                        dim=0), exact=True)
            
    def get_tokenize(self, docs):
        sent_tokens = [
            torch.cat(
                [self.tokenizer.encode(
                        sentences[i],
                        add_special_tokens=True,
                        max_length=128,
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
        
        return features
    
    def build_graph(self, features_list, tdms):
        '''get edge_index for GATLayer'''
        edge_index_list, edge_attr_list = [], []
        for features, tdm in zip(features_list, tdms):
            features = features.cpu()
            cosine_matrix = 1 - pairwise_distances(tdm, metric="cosine")

            G = nx.from_numpy_matrix(cosine_matrix)
            
            # Graph to SparseMatrix
            G = nx.to_scipy_sparse_matrix(G)
            # sparse Matrix to Graph
            edge_index, edge_attr = from_scipy_sparse_matrix(G)
            
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        return edge_index_list, edge_attr_list
    
    def gat_dataloader(self, features_list, edge_index_list, edge_attr_list, labels_list, batch_size):
        data_list = [
            torch_geometric.data.Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
                for features, edge_index, edge_attr, labels in zip(features_list, edge_index_list, edge_attr_list, labels_list)
        ]
        
        if self.use_gdc:
            data_list = [self.gdc(data) for data in data_list]
        
        gat_loader = torch_geometric.data.DataLoader(data_list, batch_size=batch_size, shuffle=False)
        return gat_loader
    
    def forward(self, 
                docs, 
                offsets,
                tdms,
                labels_list, 
                threshold=0.2, 
                batch_size=32):
        
        sent_tokens = self.get_tokenize(docs).to(DEVICE)
        word_vecs = self.embedder(sent_tokens)
        features_list = self.get_sentence_embedding(word_vecs, offsets)
        edge_index_list, edge_attr_list = self.build_graph(features_list, tdms)
        
        # dataloader for GATLayer
        dataloader = self.gat_dataloader(features_list, edge_index_list, edge_attr_list, labels_list, batch_size)
        
        output = self.gat_classifier(next(iter(dataloader)))
        return output

############################  BasicSummarizerWithGDC   ############################

class Summarizer(nn.Module):
    
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 num_heads, 
                 num_classes=2):
        super(Summarizer, self).__init__()
        
        albert_base_configuration = AlbertConfig(
            hidden_size=256,
            num_attention_heads=4,
            intermediate_size=1024,
        )
        
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.embedder = AlbertModel(albert_base_configuration)
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
            features = features.cpu()
            
            if features.ndim == 1:
                features = features.unsqueeze(0)
                
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
        
        sent_tokens = self.get_tokenize(docs).to(DEVICE)
        word_vecs = self.embedder(sent_tokens)[0]
        features_list = self.get_sentence_embedding(word_vecs, offsets)
        edge_index_list = self.build_graph(features_list, threshold)
        
        # dataloader for GATLayer
        dataloader = self.gat_dataloader(features_list, edge_index_list, labels_list, batch_size)
        
        output = self.gat_classifier(next(iter(dataloader)))
        return output