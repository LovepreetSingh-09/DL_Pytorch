# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:00:42 2019

@author: user
"""

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
%matplotlib inline

transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),
                              transforms.Normalize([0.456, 0.457, 0.455], [0.224, 0.222, 0.206])])

train=torchvision.datasets.ImageFolder('Datasets/DogsCats/train/',transform)
print(train[0][0].size())
valid=torchvision.datasets.ImageFolder('Datasets/DogsCats/valid/',transform)

train_loader=torch.utils.data.DataLoader(train,batch_size=10,shuffle=True,num_workers=0)
valid_loader=torch.utils.data.DataLoader(valid,batch_size=10,shuffle=True,num_workers=0)
print(len(train_loader))
sample_data=next(iter(train_loader))
print(np.prod(sample_data[0].size()[1:])) # torch.Size([64, 1, 28, 28])

vgg=torchvision.models.vgg16(pretrained=True)
vgg
vgg.classifier
vgg.features

if torch.cuda.is_available():
    is_cuda=True
else:
    is_cuda=False
    
# By default requires_grad is True 
for params in vgg.features.parameters():
    print(params.requires_grad)
    params.requires_grad=False
params.requires_grad

print(vgg.classifier[6].out_features)
vgg.classifier[6].out_features=2

if is_cuda:
    vgg.cuda()

optimizer=torch.optim.SGD(vgg.classifier.parameters(),lr=0.0001,momentum=0.5)

def fit(epoch,model,dataloader,optimizer,phase='training'):
    if phase=='training':
        model.train()
    else:
        model.eval()
    torch.cuda.empty_cache()
    running_loss=0.0
    running_correct=0
    for idx, (data,label) in enumerate(dataloader):
#        if is_cuda:
#            data,label=data.cuda(),label.cuda()
        input_,target=torch.autograd.Variable(data),torch.autograd.Variable(label)
        torch.cuda.empty_cache()
        output=model(input_)
        pred=output.data.max(dim=1,keepdim=True)[1]
        loss=torch.nn.functional.nll_loss(output,target)
        if phase=='training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss+=torch.nn.functional.nll_loss(output,target,size_average=False).item()
        running_correct+=pred.eq(target.data.view_as(pred)).cpu().sum().item()
        torch.cuda.empty_cache()
    f_loss=running_loss/len(dataloader.dataset)
    accuracy=running_correct/len(dataloader.dataset)
    print('Epoch ' +str(epoch) +'---Loss of '+str(phase) + ' is '+str(round(loss,4)) +' accuracy = '+ str(round(accuracy,3)))
    return f_loss,accuracy
    

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,3):
    epoch_loss, epoch_accuracy = fit(epoch,model=vgg,dataloader=train_loader,optimizer=optimizer,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model=vgg,dataloader=valid_loader,optimizer=optimizer,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)    

for layer in vgg.classifier.children():
    if(type(layer) == nn.Dropout):
        layer.p = 0.2
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,3):
    epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    
# Data Augmentation :-
train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip()
                                       ,transforms.RandomRotation(0.2) ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('Datasets/DogsCats/train/',train_transform)
valid = ImageFolder('Datasets/DogsCats/valid/',transform)

vgg=torchvision.models.vgg16(pretrained=True)
#if is_cuda:
#    vgg.cuda()

vgg_conv=vgg.features
train_loader=torch.utils.data.DataLoader(train,batch_size=32,shuffle=False,num_workers=0)
test_loader=torch.utils.data.DataLoader(valid,batch_size=32,shuffle=False,num_workers=0)

def preconv(model,dataloader):
    conv_features=[]
    labels_list=[]
    for idx,(data,label) in enumerate(dataloader):
#        if is_cuda:
#            data,label=data.cuda(),label.cuda()
        data,target=Variable(data),Variable(label)
        output=model(data)
        conv_features.extend(output)
        labels_list.extend(target)
    print(len(conv_features))
    conv_features=np.concatenate([[feat] for feat in conv_features])
    return conv_features,labels_list

conv_feat_train,labels_train = preconv(vgg_conv,train_loader)
conv_feat_valid,labels_valid = preconv(vgg_conv,test_loader)

class My_dataset(Dataset):
    def __init__(self,feat,labels):
        self.conv_feat = feat
        self.labels = labels
    
    def __len__(self):
        return len(self.conv_feat)
    
    def __getitem__(self,idx):
        return self.conv_feat[idx],self.labels[idx]

train_feat_dataset=My_dataset(conv_feat_train,labels_train)
valid_feat_dataset=My_dataset(conv_feat_valid,labels_valid)

train_feat_loader = DataLoader(train_feat_dataset,batch_size=64,shuffle=True)
val_feat_loader = DataLoader(val_feat_dataset,batch_size=64,shuffle=True)

def data_gen(conv_feat,labels,batch_size=64,shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0,len(conv_feat),batch_size):
        yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])

train_batches = data_gen(conv_feat_train,labels_train)
val_batches = data_gen(conv_feat_val,labels_val)

optimizer = optim.SGD(vgg.classifier.parameters(),lr=0.0001,momentum=0.5)

def fit_numpy(epoch,model,data_loader,phase='training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    for batch_idx , (data,target) in enumerate(data_loader):
        if is_cuda:
            data,target = data.cuda(),target.cuda()
        data , target = Variable(data,volatile),Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        data = data.view(data.size(0), -1)
        output = model(data)
        loss = F.cross_entropy(output,target)
        
        running_loss += F.cross_entropy(output,target,size_average=False).data[0]
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss,accuracy


%%time
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit_numpy(epoch,vgg.classifier,train_feat_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit_numpy(epoch,vgg.classifier,val_feat_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

class LLLLayerActivations():
    features=None
    
    def __init__(self,model,layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
    
    def hook_fn(self,module,input,output):
        self.features = output.cpu().data.numpy()
    
    def remove(self):
        self.hook.remove()
        

conv_out = LayerActivations(vgg.features,0)

o = vgg(Variable(img.cuda()))

conv_out.remove()

act = conv_out.features

fig = plt.figure(figsize=(20,50))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12,5,i+1,xticks=[],yticks=[])
    ax.imshow(act[0][i])

cnn_weights = vgg.state_dict()['features.0.weight'].cpu()

fig = plt.figure(figsize=(30,30))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12,6,i+1,xticks=[],yticks=[])
    imshow(cnn_weights[i])



