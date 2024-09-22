"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import pandas as pd
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)
from model import Net2d_1,Net2d_2




################################################################
# configs
################################################################
#该文件中有5个数据，分别是Kcoeff、Kcoeff_x、Kcoeff_y、coeff、sol维数均为1024*421*421
TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'

TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'

ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs1 = 200
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 5
h = int(((421 - 1) / r) + 1)#85
s = h#分辨率为85

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]#(1000,85,85)
y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]#(1000,85,85)

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]#(100,85,85)
y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]#(100,85,85)

#对x_train、y_train、x_test、y_test中的数据进行标准化
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)#(1000,85,85),对y_train进行标准化，训练过程中需要将y_train反标准化。没有对y_test进行标准化，训练过程不需要对y_test进行反标准化

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T#将这两个一维数组转换成了二维数组,(85*85,2),每一行代表一个点的坐标(x,y)
grid = grid.reshape(1, s, s, 2)#(1,85,85,2)，第一个channel存储的是85*85网格上x的坐标值，第二个channel存储的是y的坐标值
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)#(1000,85,85,3),1000个样本，第一个channel存储的是该点处的系数，第二个channel存储的是该点处的x坐标的值，第三个channel存储的是该点处y坐标的值
x_test = torch.cat([x_test.reshape(ntest, s, s, 1), grid.repeat(ntest, 1, 1, 1)], dim=3)#(100,85,85,3)

train_loader_1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=False)
test_loader_1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
train_error_grid1=[]
model1 = Net2d_1(modes, width).cuda()
print("模型的参数总量为：",model1.count_params())

optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
myloss1 = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs1):
    model1.train()
    train_l2=0
    for x, y in train_loader_1:
        x, y = x.cuda(), y.cuda()
        optimizer1.zero_grad()
        out1 = model1(x)
        out1 = y_normalizer.decode(out1)
        y = y_normalizer.decode(y)
        loss = myloss1(out1.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer1.step()
        train_l2 += loss.item()
    scheduler1.step()
    model1.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader_1:
            x, y = x.cuda(), y.cuda()
            out1 = model1(x)
            out1 = y_normalizer.decode(out1)
            test_l2 += myloss1(out1.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest
    train_error_grid1.append(train_l2)
    print("训练轮数：",ep,"训练集的l2相对误差：", train_l2, "测试集的l2相对相对误差：",test_l2)
    #308 2.4847933291457593 0.004443896889686584 0.00678525410592556
    #499 2.606547320727259 0.003986305944621563 0.006447487324476242

################################################################
#2,计算残差r^{1}(x)^{1},将r^{1}(x)^{1}、(x)^{1}转移到grid2上
################################################################
i=1
r_grid1_x_grid1=torch.zeros_like(y_train)#(1000,85,85)
with torch.no_grad():  # 防止梯度追踪
    for x, y in train_loader_1:
        x, y = x.cuda(), y.cuda()
        out1 = model1(x)
        out1 = y_normalizer.decode(out1)
        y = y_normalizer.decode(y)
        r_grid1_x_grid1[(i - 1) * batch_size:(i * batch_size), :,:]=y-out1.detach()
        i=i+1
i=1
y_hat_grid1=torch.zeros_like(y_train)#(1000,85,85)
with torch.no_grad():  # 防止梯度追踪
    for x, y in train_loader_1:
        x, y = x.cuda(), y.cuda()
        out1=model1(x)
        out1 = y_normalizer.decode(out1)
        y_hat_grid1[(i - 1) * batch_size:(i * batch_size), :,:]=out1.detach()
        i=i+1

#################################################################
# 3,grid2上训练NN^{2,v}
############
################################################################
ntrain = 1000
ntest = 100
batch_size = 20
learning_rate = 0.001
epochs2 = 500
step_size = 100
gamma = 0.5
modes = 12
width = 32
r = 5
h = int(((421 - 1) / r) + 1)#85
s = h#分辨率为85
reader = MatReader(TRAIN_PATH)
x_train_2 = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]#(1000,85,85)
y_train_2 = r_grid1_x_grid1
#对x_train、y_train、x_test、y_test中的数据进行标准化
x_normalizer_2 = UnitGaussianNormalizer(x_train_2)
x_train_2 = x_normalizer_2.encode(x_train_2)
y_normalizer_2 = UnitGaussianNormalizer(y_train_2)

y_train_2 = y_normalizer_2.encode(y_train_2)#(1000,85,85),对y_train进行标准化，训练过程中需要将y_train反标准化。没有对y_test进行标准化，训练过程不需要对y_test进行反标准化

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T#将这两个一维数组转换成了二维数组,(85*85,2),每一行代表一个点的坐标(x,y)
grid = grid.reshape(1, s, s, 2)#(1,85,85,2)，第一个channel存储的是85*85网格上x的坐标值，第二个channel存储的是y的坐标值
grid = torch.tensor(grid, dtype=torch.float)
x_train_2 = torch.cat([x_train_2.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)#(1000,85,85,3),1000个样本，第一个channel存储的是该点处的系数，第二个channel存储的是该点处的x坐标的值，第三个channel存储的是该点处y坐标的值
x_train_2=torch.cat([x_train_2,y_hat_grid1.unsqueeze(-1)],dim=3)
train_loader_2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_2, y_train_2), batch_size=batch_size,shuffle=False)
################################################################
# training and evaluation
################################################################
model2 = Net2d_2(modes, width).cuda()
train_error_grid2=[]
print("模型的参数总量为：",model2.count_params())

optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)
myloss2 = LpLoss(size_average=False)
y_normalizer_2.cuda()

for ep in range(epochs2):
    model2.train()
    train_l2=0
    for x, y in train_loader_2:
        x, y = x.cuda(), y.cuda()
        optimizer2.zero_grad()
        out2 = model2(x)
        out2 = y_normalizer_2.decode(out2)
        y = y_normalizer_2.decode(y)
        loss = myloss2(out2.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer2.step()
        train_l2 += loss.item()
    scheduler2.step()
    model1.eval()
    model2.eval()
    test_l2 = 0.0

    with torch.no_grad():
        for (x1, y1),(x2,y2) in zip(train_loader_1,train_loader_2):
            x1, y1 = x1.cuda(), y1.cuda()
            x2,y2=x2.cuda(),y2.cuda()
            out1 = model1(x1)
            out1 = y_normalizer.decode(out1)
            out2=model2(x2)
            out2 = y_normalizer_2.decode(out2)
            out_sum=out1+out2
            y1=y_normalizer.decode(y1)
            test_l2 += myloss1(out_sum.view(batch_size, -1), y1.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntrain
    train_error_grid2.append(test_l2)
    print("训练轮数：",ep,"训练集的l2相对误差：", train_l2, "测试集的l2相对相对误差：",test_l2)
train_error_grid1+=[np.nan]*300
data={
    'gird1':train_error_grid1,
    'grid2':train_error_grid2
}
df=pd.DataFrame(data)
df.to_excel('darcy2.xlsx',index=False)


