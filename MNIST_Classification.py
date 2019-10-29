# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:31:26 2019

@author: user
"""

import numpy as np
import torch
from torchvision import datasets, transforms
from torch import optim
import torchvision
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
%matplotlib inline

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train=datasets.MNIST('Datasets/',transform=transform,train=True,download=False)
test=datasets.MNIST('Datasets/',transform=transform,train=False,download=False)

train_loader=torch.utils.data.DataLoader(train,batch_size=20,shuffle=True);len(train_loader)
test_loader=torch.utils.data.DataLoader(test,batch_size=20,shuffle=True);len(test_loader)

sample_data=next(iter(train_loader))
print(np.prod(sample_data[0].size()[1:])) # torch.Size([64, 1, 28, 28])

def imshow(img):
    print(img.shape)
    print(img.numpy().shape)
    img=img.numpy()[0]
    print(img.shape)
    mean=0.1307
    std=0.3081
    img = img*std + mean
    plt.imshow(img,cmap='gray')
    
imshow(sample_data[0][0])
imshow(sample_data[0][1])

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)
    
    def forward(self,x):
        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x=x.view(-1,320)
        x=self.fc1(x)
        x=F.dropout(x,p=0.2,training=self.training)
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)

model=Net()
if torch.cuda.is_available():
    model.cuda()
model
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
data,target=next(iter(train_loader))

if torch.cuda.is_available():
    data,target=Variable(data.cuda()),Variable(target.cuda())
else:
    data,target=Variable(data),Variable(target)
 
output=model(data)
print(output.size()) 
print(data.size())
print(target.size());target
# max function returns max values with max index 
preds=output.data.max(dim=1,keepdim=True)[1];preds
r=preds.eq(target.data.view_as(preds)).cpu().sum().numpy()
r/50
loss=F.nll_loss(output,target).data;loss
loss/5
f=torch.sum(preds==target.data).cpu().numpy()
f/5

def fit(epoch,model,dataloader,optimizer,phase='training'):
    if phase=='training':
        model.train()
    if phase=="validation":
        model.eval()
    
    running_loss=0.0
    running_correct=0
    for batch_idx, (train,target) in enumerate(dataloader):
        if torch.cuda.is_available():
            data=Variable(train.cuda())
            target=Variable(target.cuda())
        else:
            data,target=Variable(data),Variable(target)
        if phase=='training':
            optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        running_loss+=F.nll_loss(output,target,size_average=False).item()
        preds=output.data.max(dim=1,keepdim=True)[1]
        running_correct+=preds.eq(target.data.view_as(preds)).cpu().sum().numpy()
        if phase=='training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(dataloader.dataset)
    accuracy = 100.0 * running_correct/len(dataloader.dataset)
    print('Epoch ' +str(epoch) +'---Loss of '+str(phase) + ' is '+str(round(loss,4)) +' accuracy = '+ str(round(accuracy,3)))
    
    return loss,accuracy

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_loader,optimizer,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_loader,optimizer,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)    
      

train_loader.dataset

plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()
plt.show()

plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'bo',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
plt.legend()
plt.show()

