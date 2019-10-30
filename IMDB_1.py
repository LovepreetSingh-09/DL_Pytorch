# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:31:19 2019

@author: user
"""

import torchtext
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import optim
import torchvision
from torch.autograd import Variable
from glob import glob
from torch import nn
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe

is_cuda = False
if torch.cuda.is_available():
    is_cuda=True
    
help(torchtext.data.Field)
TEXT=torchtext.data.Field(lower=True,batch_first=True,fix_length=40)
LABEL=torchtext.data.Field(sequential=False)

train,test=torchtext.datasets.IMDB.splits(TEXT,LABEL)
type(train)
train.fields
len(train)
help(vars)
vars(train[0])

TEXT.build_vocab(train,vectors=torchtext.vocab.GloVe(name='6B',dim=50),max_size=10000,min_freq=10)
LABEL.build_vocab(train)

TEXT.vocab
TEXT.vocab.freq
TEXT.vocab.vectors
TEXT.vocab.stoi

train_iter,test_iter=torchtext.data.BucketIterator.splits((train,test),batch_size=128,device=None,shuffle=True)
train_iter.repeat=False
test_iter.repeat=False
next(iter(train_iter))
b=set()

vars(train[0])['text']
for i in range(25000):
    for word in vars(train[i])['text']:
        b.add(word)
len(b)

class EmbedNet(torch.nn.Module):
    def __init__(self,n_words,embed_dim,hidden=400):
        super().__init__()
        self.embedding=torch.nn.Embedding(n_words,embed_dim)
        self.fc=torch.nn.Linear(hidden,3)
    def forward(self,x):
        x=self.embedding(x).view(x.size(0),-1)
        x=self.fc(x)
        return torch.nn.functional.log_softmax(x,dim=-1)
    
model = EmbedNet(len(TEXT.vocab.stoi),10)
model = model.cuda()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def fit(epoch,model,data_loader,phase='training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
    running_loss = 0.0
    running_correct = 0
    for batch_idx , batch in enumerate(data_loader):
        text , target = batch.text , batch.label
        if is_cuda:
            text,target = text.cuda(),target.cuda()
        
        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output,target)
        running_loss += F.nll_loss(output,target,size_average=False).item()
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().item()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss,accuracy

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

for epoch in range(1,10):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

#Using pretrained Glove word embeddings
TEXT = torchtext.data.Field(lower=True, batch_first=True,fix_length=40,)
LABEL = torchtext.data.Field(sequential=False,)

TEXT.build_vocab(train,test, vectors=GloVe(name='6B', dim=300),max_size=10000,min_freq=10)
LABEL.build_vocab(train,)

class EmbNet(nn.Module):
    def __init__(self,n_words,embed_dim,hidden=400):
        super().__init__()
        self.embedding=torch.nn.Embedding(n_words,embed_dim)
        self.fc=torch.nn.Linear(hidden,3)
    def forward(self,x):
        x=self.embedding(x).view(x.size(0),-1)
        x=self.fc(x)
        return torch.nn.functional.log_softmax(x,dim=-1)

model = EmbNet(len(TEXT.vocab.stoi),300,12000)
model = model.cuda()

model.embedding.weight.data = TEXT.vocab.vectors.cuda()

model.embedding.weight.requires_grad = False

#optimizer = optim.SGD(model.parameters(),lr=0.001)
optimizer = optim.Adam([ param for param in model.parameters() if param.requires_grad == True],lr=0.001)
for epoch in range(1,10):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)


