import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import time

os.getcwd()
path = 'Datasets/DogsCats/'
files = glob(os.path.join(path, '*/*/*.jpg'))
n_images = len(files)
print('No. of images : {}'.format(n_images))

shuffle = np.random.permutation(n_images)

for t in ['train', 'valid']:
    for c in ['Dog', 'Cat']:
        os.makedirs(os.path.join(path, t, c))


for idx in shuffle[:3000]:
    image = files[idx].split('\\')[-1]
    folder = files[idx].split('\\')[-2]
    os.rename(files[idx], os.path.join(path, 'valid', folder, image))

for idx in shuffle[3000:]:
    image = files[idx].split('\\')[-1]
    folder = files[idx].split('\\')[-2]
    os.rename(files[idx], os.path.join(path, 'train', folder, image))

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(), transforms.Normalize([0.456, 0.457, 0.455], [0.224, 0.222, 0.206])])

train = ImageFolder('Datasets/DogsCats/train/', transform)
valid = ImageFolder('Datasets/DogsCats/valid/', transform)
print(train.class_to_idx)
print(train.classes)


def imshow(img):
    print(img.shape)
    img = img.numpy().transpose((1, 2, 0))
    print(img.shape)
    mean, std = [0.456, 0.457, 0.455], [0.224, 0.222, 0.206]
    img = std*img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)


print(len(train[50][0]))
imshow(train[50][0])

train_gn = torch.utils.data.DataLoader(
    train, shuffle=True, batch_size=16, num_workers=0)
valid_gn = torch.utils.data.DataLoader(valid, batch_size=16, num_workers=0)

dataset_size = {'train': len(train_gn.dataset),
                'valid': len(valid_gn.dataset)}
dataloaders = {'train': train_gn, 'valid': valid_gn}

model = models.resnet18(pretrained=True)
n_features = model.fc.in_features
print(n_features)
model.fc = nn.Linear(n_features, 2)
model.state_dict()

if torch.cuda.is_available():
    model.cuda()

model

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model,criterion,optimizer,scheduler,n_epochs=10):
    tic=time.time()
    best_model_wts=model.state_dict
    best_acc=0.0
    for epoch in range(n_epochs):
        start=time.time()
        print('{} / {}'.format(epoch+1,n_epochs))
        print('-'*10)
        torch.cuda.empty_cache()

        for ph in ['train','valid']:
            if ph=='train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            
            running_loss=0.0
            correct=0
            b=0
            for data in dataloaders[ph]:
                inputs,labels=data
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    inputs= Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs=Variable(inputs)
                    labels=Variable(labels)
#                inputs=Variable(inputs)
#                labels=Variable(labels)
                optimizer.zero_grad()
                output=model(inputs)
                _,pred=torch.max(output,1)
                loss=criterion(output,labels)
                b+=16
                if b%220==0:
                    print('#',end=' ')
                if ph=='train':
                    loss.backward()
                    optimizer.step()     
                running_loss+=loss.item()
                correct+=torch.sum(pred==labels.data)
            print()
            epoch_loss=running_loss/dataset_size[ph]
            epoch_acc=correct/dataset_size[ph]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(ph, epoch_loss, epoch_acc))
            end=time.time()
            time_=end-start
            print('Training complete in {:.0f}m {:.0f}s'.format(time_ // 60, time_ % 60))
            if ph=='valid' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=model.state_dict()
        print()
    time_elapsed = time.time() - tic
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model=train_model(model,criterion,optimizer,exp_lr_scheduler,n_epochs=2)

torch.cuda.memory_allocated()
torch.cuda.empty_cache()
