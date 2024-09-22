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

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


# Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)#(20,32,85,43,2)对最后两个维度进行FFT操作，最后一个维度为2代表了傅里叶变换后的实部和虚部

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)#(20,32,85,43,2)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        return x
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1#12
        self.modes2 = modes2#12
        self.width = width#32
        self.padding = 9  # pad the domain if input is non-periodic
        self.p = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        batchsize = x.shape[0]#20
        size_x, size_y = x.shape[1], x.shape[2]#85,85

        x = self.p(x)#fc0():nn.Linear(3, self.width), x:(20,85,85,3)->(20,85,85,32)
        x = x.permute(0, 3, 1, 2)#重新排列维度,x:(20,85,85,32)->(20,32,85,85)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)#
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2+ x
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2+x
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2+x
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2+x

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


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

epochs1 = 100
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
model1 = Net2d(modes, width).cuda()
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
#################################################################
# 3,grid2上训练NN^{2,v}
############
################################################################
ntrain = 1000
ntest = 100
batch_size = 20
learning_rate = 0.00025
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
train_loader_2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_2, y_train_2), batch_size=batch_size,shuffle=False)
################################################################
# training and evaluation
################################################################
model2 = Net2d(modes, width).cuda()
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
        for x, y in train_loader_1:
            x, y = x.cuda(), y.cuda()
            out1 = model1(x)
            out1 = y_normalizer.decode(out1)
            out2=model2(x)
            out2 = y_normalizer_2.decode(out2)
            out_sum=out1+out2
            y=y_normalizer.decode(y)
            test_l2 += myloss1(out_sum.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntrain
    #test_l2 /= ntest
    print("训练轮数：",ep,"训练集的l2相对误差：", train_l2, "测试集的l2相对相对误差：",test_l2)