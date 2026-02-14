---
date : '2026-02-14T19:24:57+08:00'
draft : false
title : 'Softmax回归'
tags: ['d2l','deeplearning']
categories: []
math: true
---
**abstract: 李沐动手深度学习--softmax回归的数学原理与从零实现**
**story**: 什么是适合新手的教程？
**Attention: 关于语法的讲解主要涉及torch库，其他库的用法可能略有出入**

# 一. 回归 vs 分类
**本质上：输出连续与否**
譬如softmax回归，其输出是**预测为某类别的概率**，是连续值，属于[0, 1]之间，故即使最后用于分类问题，其本质也是一个回归模型

# 二. 神经元与激活函数
![neural](/image/neural.png)
## 2.1 神经元
所谓神经元，指的是一个运算过程：`f(W*X + b)` ，`W`是该神经元前一层的权重，`X`是该神经元前一层的输出，这两者通常是两个矩阵。b是偏置，决定W*X有多难被激活（对于阶跃函数或ReLu这种激活函数，自变量小于零时函数值为0，故可以通过调节偏置b的值，进而调节W\*X被激活的阈值）。而f()就是激活函数。
## 2.2 激活函数
神经网络中，激活函数必须是非线性的，否则网络的层数和深度都没有意义。而激活函数是非线性的，不代表网络一定是非线性的。非线性的网络需要满足两个条件：
- 非线性的激活函数
- 复合运算或者高次运算<br>
在网络中，**复合运算**就是指**多层网络**，即至少存在一层隐藏层神经元，两组W权重矩阵的网络

# 三. softmax
## 3.1 两个作用
- 非负化
- 将数值映射到0-1区间（概率化）

## 3.2 公式
$$\frac{}$$


# 四.代码一般结构
## 4.1 工具函数
### 4.1.1 定义net
```python
def softmax(X):
    '''
    softmax 的 Docstring
    
    :param X: 二维矩阵，每行代表一个样本，每列代表一个类别，即每行代表一个样本的各类别的得分
    '''
    X_exp = torch.exp(X)
    example_sum = X_exp.sum(dim=1, keepdim=True)
    return X_exp / example_sum

def net(X): 
    '''
    net 的 Docstring
    batch_size*(H*W) @ (H*W)*num_classes -> batch_size*num_classes
    :param X: 输入数据，形状为 (batch_size, 1, 28, 28)，需要先 reshape 成 (batch_size, 784) 才能与权重矩阵 W 相乘.
    
    '''
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]), W) + b)
    # 这里reshape不会改变原始数据形状, 也不创建新副本， 数据还是那个数据，这里创建了一个新视图，改变了查看数据的方式

```
-`sum(dim = 0)`: dim等于几，shape中第几项就变为1。比如dim = 0，一个[2,3]的矩阵变为了[1,3]，也就是对每一行求和，成了一个列向量；同样地，dim = 1时对每一列求和，成了一个行向量
- `torch.matmul()`: 等效于`A@B`
- X为[256, 1, 28, 28]形式，这里以[256, 784]形状载入，也就是[256, 784] * [784, 10] + [10, ]--> [256, 10]


### 4.1.2 定义loss
```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-9)
```
- 
- 加一个极小的数防止除数为零
## 4.2 主函数
### 4.2.1 导入并定义数据集
#### 导入库就不说了
```python
import torch
import numpy as np
import random
import torchvision
```
#### 超参数定义在一起 之后好调参
```python
input_size = 784
num_classes = 10 
batch_size = 256
num_epochs = 20
lr = 0.1 # 线性模型对softmax回归时，最好设置lr随着训练轮数增大而衰减，否则会出现loss波动
```
#### 这里使用fashionMNIST数据集，torch封装好了函数，直接完成导入 + 定义
```python
trans = torchvision.transforms.ToTensor() 
mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=trans)
mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=trans)
```
- `trans`:三个作用
    1.转化为tensor 
    2.归一化到[0,1] 
    3.形状变为channel*H*W

- 参数`download`: 如果没有下载数据集，则自动下载 
- 可以继承`Dataset`类，自定义数据集结构，也就是设计一个样本的数据以什么形式载入模型，如HSI攻击中将数据集设计为六个张量。

### 4.2.2 定义dataloader
```python
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)
```
- 将整个数据集划分为batch
- 每个batch含有256个样本，也就是X, y两个张量
    -`X`: [256, 1, 28, 28]-[样本数, channel, width, height]
    -`y`: [256] 这256个样本对应的标签
- 参数`shuffle`: 训练时打乱样本顺序，增强鲁棒性   

### 4.2.3 定义参数
```python
# W = torch.randn((input_size, num_classes), requires_grad=True)  
#只能生成高斯分布均值为0，方差为1的随机数，不能指定均值和方差
W = torch.normal(0, 0.01, size=(input_size, num_classes), requires_grad=True)
b = torch.zeros((num_classes,), requires_grad=True)
```
- `requires_grad`: 参数的标志，表明对这个变量求梯度。有两种书写办法：
    - 一是像这里一样，初始化时使用requires_grad参数
    - 二是使用`a.requires_grad = True`属性

### 4.2.4 train过程
```python
loss = cross_entropy
for epoch in range(num_epochs):
    train_loss, n = 0, 0
    for X, y in train_loader:
        # print(f"X_shape_init: {X.shape}") # torch.Size([256, 1, 28, 28])
        y_hat = net(X)
        l = loss(y_hat, y)
        # print(f"y shape: {y.shape}") # torch.Size([256])
        # print(f"loss shape: {l.shape}") # torch.Size([256])
        (l.sum() / X.shape[0]).backward() 
        train_loss += l.sum().item()
        n += X.shape[0]
        with torch.no_grad():
            lr_real = lr * (0.90 ** epoch)
            W -= lr_real*W.grad # -=:原地修改  W = W - lr*W.grad:产生新的W
            b -= lr_real*b.grad
            W.grad.zero_()
            b.grad.zero_()
    print(f"Epoch {epoch+1}, Loss: {(train_loss / n)}")
```
**Attention：**
- 手动更新参数时要注意切换到`torch.no_grad()`状态。否则会被认为对参数进行了运算，该参数不再是叶子张量，不能查看`grad`属性
- 每一轮迭代的梯度不能累加，否则会梯度爆炸
- 反向传播时对**平均损失**进行，即所有样本的损失的均值，不要对损失之和反向传播
- 求loss的时候注意求全局loss

### 4.2.5 test过程
···python
correct, total, loss_total = 0, 0, 0
for X, y in test_loader: 
    y_hat = net(X)
    loss_total += loss(y_hat, y).sum().item()
    correct += (y_hat.argmax(dim=1).type(y.dtype) == y).sum().item()
    total += y.shape[0]  
print(f"Accuracy: {correct / total} Test loss: {loss_total / total}")
···
- `a.type(B)`:将a转换为B类型 或 a.to(B)
- `a.dtype`:查看a的数据类型

## 4.3 完整代码
```python
import torch
import numpy as np
import random
import torchvision

input_size = 784
num_classes = 10 
batch_size = 256
num_epochs = 20
lr = 0.1

# W = torch.randn((input_size, num_classes), requires_grad=True)  
#只能生成高斯分布均值为0，方差为1的随机数，不能指定均值和方差
W= torch.normal(0, 0.01, size=(input_size, num_classes), requires_grad=True)
b = torch.zeros((num_classes,), requires_grad=True)

trans = torchvision.transforms.ToTensor() 
# 1.转化为tensor 2.归一化到[0,1] 3.形状变为channel*H*W
mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=trans)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)

mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=trans)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)

def softmax(X):
    '''
    softmax 的 Docstring
    
    :param X: 二维矩阵，每行代表一个样本，每列代表一个类别，即每行代表一个样本的各类别的得分
    '''
    X_exp = torch.exp(X)
    example_sum = X_exp.sum(dim=1, keepdim=True)
    return X_exp / example_sum

def net(X): 
    '''
    net 的 Docstring
    batch_size*(H*W) @ (H*W)*num_classes -> batch_size*num_classes
    :param X: 输入数据，形状为 (batch_size, 1, 28, 28)，需要先 reshape 成 (batch_size, 784) 才能与权重矩阵 W 相乘.
    
    '''
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]), W) + b)
    # 这里reshape不会改变原始数据形状, 也不创建新副本， 数据还是那个数据，这里创建了一个新视图，改变了查看数据的方式

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-9)

 
loss = cross_entropy
for epoch in range(num_epochs):
    train_loss, n = 0, 0
    for X, y in train_loader:
        # print(f"X_shape_init: {X.shape}") # torch.Size([256, 1, 28, 28])
        y_hat = net(X)
        l = loss(y_hat, y)
        # print(f"y shape: {y.shape}") # torch.Size([256])
        # print(f"loss shape: {l.shape}") # torch.Size([256])
        (l.sum() / X.shape[0]).backward() 
        train_loss += l.sum().item()
        n += X.shape[0]
        with torch.no_grad():
            lr_real = lr * (0.90 ** epoch)
            W -= lr_real*W.grad # -=:原地修改  W = W - lr*W.grad:产生新的W
            b -= lr_real*b.grad
            W.grad.zero_()
            b.grad.zero_()
    print(f"Epoch {epoch+1}, Loss: {(train_loss / n)}")
        
correct, total, loss_total = 0, 0, 0
for X, y in test_loader: 
    y_hat = net(X)
    loss_total += loss(y_hat, y).sum().item()
    correct += (y_hat.argmax(dim=1).type(y.dtype) == y).sum().item()
    # a.type(B):将a转换为B类型 或 a.to(B)
    # a.dtype:查看a的数据类型
    total += y.shape[0]  
print(f"Accuracy: {correct / total} Test loss: {loss_total / total}")
```

# 训练结果
- lr衰减后震荡问题依然存在，可能步长还是太大了，或者控制衰减的超参数没有调好
![result](/image/result.png)
