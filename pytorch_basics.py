import autopep8
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_boston
from torch.autograd import Variable

boston = load_boston()
boston.keys()
x = torch.rand(10)
x.size()
temp = torch.FloatTensor([0.1, 4.3, 5.2, 6.1, 5.2, .34, 2.6, 7.3])
temp.size()

bos_tensor = torch.from_numpy(boston.data)
bos_tensor.size()
bos_tensor[:2]

Image.open('Files//171846.jpg').resize((224, 224))
pic = np.array(Image.open('Files//171846.jpg').resize((224, 224)))
pic
torch_pic = torch.from_numpy(pic)
torch_pic
torch_pic.size()
plt.imshow(torch_pic)

plt.imshow(torch_pic[:, :, 0])
plt.imshow(torch_pic[:, 0:100, :])
plt.imshow(torch_pic[:, 0:100, :].numpy())

x = Variable(torch.ones(2, 2), requires_grad=True)
x.data
y = x.mean()
print(y)
x.grad.data *= 0
x.grad.data
y.backward()
x.grad
y.grad_fn

a = torch.rand(5, 5)
b = torch.rand(5, 5)
c = a+b
d = torch.add(a, b)
d
a*b
a.matmul(b)
a.mul(b)
a.mul_(b)

a = torch.rand(1000, 1000)
b = torch.rand(1000, 1000)
a.matmul(b)

a = a.cuda()
b = b.cuda()
a.matmul(b)
