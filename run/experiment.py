import json
from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader, random_split

from models import BasicSummarizer
from types_ import *
from .dataset import SummaryDataset


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class SummaExperiment(pl.LightningModule):
    
    def __init__(self,
                 model: BasicSummarizer,
                 params: dict) -> None:
        
        super(SummaExperiment, self).__init__()
        
        self.model = model
        self.params = params
        # self.curr_device = None
        
        
    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, docs, offsets, labels) -> Tensor:
        return self.model(docs, offsets, labels)
    
    def loss_function(self, logits, labels):
        labels = torch.cat(
            [torch.tensor(label, dtype=torch.float) for label in labels]
        )
        labels = labels.view(-1, logits.size()[1]).to(DEVICE)
        logits = logits.view(-1, logits.size()[1])
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return bce_loss
    
    def accuracy(self, logits, labels):
        """Computes the accuracy for multiple binary predictions"""

        labels = torch.cat(
            [torch.tensor(label, dtype=torch.float) for label in labels]
        )
        labels = labels.view(-1, logits.size()[1]).to(DEVICE)
        logits = logits.view(-1, logits.size()[1])

        preds = torch.round(logits)
        corrects = (preds == labels).sum().float()
        acc = corrects / labels.numel()
        return acc
    
    
    def training_step(self, batch, batch_idx):
        docs, offsets, labels = batch
        
        logits = self.forward(docs, offsets, labels)
        train_loss = self.loss_function(logits, labels)
        train_acc = self.accuracy(logits, labels)
        
        tqdm_dict = {'train_acc': train_acc}
        output = OrderedDict({
            'loss': train_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output
    
    def validation_step(self, batch, batch_idx):
        docs, offsets, labels = batch
        
        logits = self.forward(docs, offsets, labels)
        val_loss = self.loss_function(logits, labels)
        
        # acc
        val_acc = self.accuracy(logits, labels)
        
        output = OrderedDict({
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        return output
    
    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
        
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result
    
    
    def test_step(self, batch, batch_idx):
        docs, offsets, labels = batch
        logits = self.forward(docs, offsets, labels)
        test_loss = self.loss_function(logits, labels)
        
        # acc
        test_acc = self.accuracy(logits, labels)
        
        output = OrderedDict({
            'test_loss': test_loss,
            'test_acc': test_acc,
        })
        return output
    
    def test_epoch_end(self, outputs):
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss_mean += output['test_loss']
            test_acc_mean += output['test_acc']
        
        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        tqdm_dict = {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'test_loss': test_loss_mean}
        return result
    
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        
        return [optimizer]
    
    
    @staticmethod
    def __collate_fn(batch):
        docs = [entry[0] for entry in batch if len(entry[0]) > 1]
        labels = [entry[1] for entry in batch if len(entry[1]) > 1]
        offsets = [0] + [len(doc) for doc in docs]
        return docs, offsets, labels
    
    def __dataloader(self, phase='train'):
        
        data_path = '../../data/summary/data/train.json'
        dataset = SummaryDataset(data_path)
        
        # split - train/valid/test
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))
        test_size = len(dataset) - (train_size + valid_size)

        train_dataset, valid_dataset, test_dataset \
            = random_split(dataset, [train_size, valid_size, test_size])
        
        if phase == 'train':
            loader =  DataLoader(train_dataset, 
                                 batch_size=self.params['batch_size'], 
                                 shuffle=True, 
                                 collate_fn=self.__collate_fn)
        elif phase == 'valid':
            loader =  DataLoader(valid_dataset, 
                                 batch_size=self.params['batch_size'], 
                                 shuffle=False, 
                                 collate_fn=self.__collate_fn)
        elif phase == 'test':
            loader =  DataLoader(test_dataset, 
                                 batch_size=self.params['batch_size'], 
                                 shuffle=False, 
                                 collate_fn=self.__collate_fn)
        
        return loader
    
    def train_dataloader(self):
        # log.info('Training data loader called.')
        print('Training data loader called.')
        return self.__dataloader(phase='train')
    
    def val_dataloader(self):
        # log.info('Validation data loader called.')
        print('Validation data loader called.')
        return self.__dataloader(phase='valid')
    
    def test_dataloader(self):
        # log.info('Test data loader called.')
        print('Test data loader called.')
        return self.__dataloader(phase='test')