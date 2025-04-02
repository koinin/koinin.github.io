+++
date = "2025-03-02"
draft = false
title = 'Note'

+++

[toc]

# TODO

- StyleGAN
- 蛋白质图片分类
- 强化学习MAPPO，betaPPO,随机策略

## GAN

### Stylegan2-ada-pytorch

遇到一个问题，训练了很久也没法降低fid

### FastGAN

> [odegeasslbc/FastGAN-pytorch: Official implementation of the paper "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis" in ICLR 2021](https://github.com/odegeasslbc/FastGAN-pytorch?tab=readme-ov-file)

对于fow-shot, high resolution 有效果

- 测试



## 蛋白质图片分类



## 强化学习

# 文献和资料

### 上采样
超分，通过插值让图片更大

```python
class DSConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernels_per_layer=4, groups=1, old=0):
        super(DSConv2d, self).__init__()
        if old == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups)
            )
        elif old == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes * kernels_per_layer),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes * kernels_per_layer),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class UNetUp(nn.Module):
# 继承自nn.module的类，然后定义其基本属性__init__()，foward()
# 演示其基本用法
def __init__(self, in_channels, residual_in_channels, out_channels, size, old=0):
        super(UNetUp, self).__init__()
        self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.conv = DSConv2d(in_channels + residual_in_channels, out_channels, 1, 1, old=old)
        # 这里初始化了基本的对象
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        # 调用了对象的前向传播
        return x
# 所以要使用此类也要先初始化对象，然后传入参数调用前向传播


```

### CNN

> [CNN模型合集 - 知乎](https://www.zhihu.com/column/ConvNets)

#### 技巧

- 小卷积核代替大卷积
- 深度VS宽度
- 1*1卷积核妙用，减少参数
- 正则化方法，BN、Dropout
- 模型优化技巧

- 感受野

#### 结构

- VGG（VGG16，VGG19）

![](https://pica.zhimg.com/70/v2-dfe4eaaa4450e2b58b38c5fe82f918c0_1440w.avis?source=172ae18b&biz_tag=Post)

- Inception

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*qOfJ2eam6zKfc_awjjFXrw.png)

- ResNet结构
- Fractal结构，分型网络的Drop-path（路径舍弃）
- dual path结构，DPN
- SE模块

#### 卷积：

- 标准卷积
- 分组卷积，通道的局部特性
- 空洞卷积，增大感受野而不损失信息
- 反卷积
- 深度可分离卷积，分为两部分Depthwise+Pointwise，很常用



## Pytorch.Lighting

学习了一点pytorch lighting 的基本使用方法

像这样把train()包装进L.LightingModule

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
  def forward(self, x):
    return self.l1(x)

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
  def forward(self, x):
    return self.l1(x)

class LitAutoEncodert(L.LightningModule):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
  def training_step(self, batch, batch_idx):
    x, _ = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x, x_hat)
    return loss
    

  def test_step(self, batch, batch_idx):
    x, _ = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    test_loss = F.mse_loss(x_hat, x)
    self.log("test_loss", test_loss)
  def validation_step(self, batch, batch_idx):
    # this is the validation loop
    x, _ = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    val_loss = F.mse_loss(x_hat, x)
    self.log("val_loss", val_loss)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer


from torchvision import datasets
import torch.utils.data as data

transform=transforms.ToTensor()
train_set = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)
test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

if __name__ == '__main__':
    autoencoder = LitAutoEncodert(Encoder(), Decoder())
    trainer = L.Trainer(max_epochs=10)
    train_loader = DataLoader(train_set)
    valid_loader = DataLoader(valid_set)
    trainer.fit(autoencoder, train_loader, valid_loader)
    trainer.test(autoencoder, dataloaders=test_loader)

# 这里我还学到了一个方法，import的时候会默认执行目标库的代码，所以要想不执行，要加一个if __name__ = '__main__':

# 调用就很简单了：
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from test import Encoder, Decoder, LitAutoEncodert

# Load the model, 这里是作为参数传进去的
model = LitAutoEncodert.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=9-step=120000.ckpt",
    encoder=Encoder(),
    decoder=Decoder()
)


# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the appropriate device
model.to(device)
model.eval()

# Load MNIST test dataset
test_dataset = MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate the model
total_loss = 0
# 不调用计算图，可以加速
with torch.no_grad():
    for data in test_loader:
        images, _ = data
        images = images.view(images.size(0), -1).to(device)  # Flatten and move to device
        encoded = model.encoder(images)
        outputs = model.decoder(encoded)
        loss = F.mse_loss(outputs, images, reduction='sum')
        total_loss += loss.item()

average_loss = total_loss / len(test_loader.dataset)
print('Average reconstruction loss on the 10000 test images: {:.4f}'.format(average_loss))

```

## Pytorch

### Model

建立一个Module只需要解决三个问题：

- Dateset
- model
- train_loop

```python
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
# 导入tqdm
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import timm


train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

class SimpleClassifier(nn.Module):
    def __init__(self, out_put_size):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, out_put_size)  # 输出是10个类
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size,32*7*7)
        output = self.out(x)
        return output
    
model = SimpleClassifier(10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(model)

num_epochs = 100
train_losses = []
valid_losses = []
# 之前梯度不更新是因为这里又创建了一个model = SimpleClassifier(10)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in tqdm(range(num_epochs), desc='Epochs', leave=True):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        batch_loss = criterion(output, labels)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item() * images.size(0)
        
    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
    # 保存模型
    torch.save(model.state_dict(), f'best.pth')
    
# Predict
# 加载模型权重
model.load_state_dict(torch.load('best.pth'))
model.to(device)

# 测试预测效果
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # 打印前20个结果
        if i == 0:  # 只打印第一个batch的前20个
            print("预测值\t真实值")
            for j in range(min(20, len(labels))):
                print(f"{preds[j].item()}\t{labels[j].item()}")
            break

```

### Tensor



## Accelerate



