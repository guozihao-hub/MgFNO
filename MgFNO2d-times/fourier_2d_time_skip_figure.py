"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)


# Complex multiplication
def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

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
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)
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

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.p = nn.Linear(12,self.width)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y), x, y)


        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)

        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x=x1+x2+x
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2 + x
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2 + x
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2 + x
        x = F.gelu(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x=self.q(x)
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
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


################################################################
# configs
################################################################
TRAIN_PATH = 'ns_V1e-3_N5000_T50.mat'
TEST_PATH = 'ns_V1e-3_N5000_T50.mat'
#TRAIN_PATH = 'ns_V1e-5_N1200_T20.mat'
#TEST_PATH = 'ns_V1e-5_N1200_T20.mat'

ntrain = 1000
ntest = 200

modes = 12
width = 20

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5



path = 'ns_fourier_2d_rnn_V10000_T20_N' + str(ntrain) + '_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/' + path
path_train_err = 'results/' + path + 'train.txt'
path_test_err = 'results/' + path + 'test.txt'
path_image = 'image/' + path

runtime = np.zeros(2, )

sub = 1
S = 64
T_in = 10
T = 10
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :T_in]#(1000,64,64,10)前10个时刻
train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, T_in:T + T_in]#(1000,64,64,10)第11个时刻到第20个时刻

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :T_in]
test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, T_in:T + T_in]


assert (S == train_u.shape[-2])#如果条件为假，抛出AssertionError异常
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain, S, S, T_in)
test_a = test_a.reshape(ntest, S, S, T_in)

# pad the location (x,y)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])#(1,64,64,1)
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])#(1,64,64,1)

train_a = torch.cat((train_a, gridx.repeat([ntrain, 1, 1, 1]), gridy.repeat([ntrain, 1, 1, 1])), dim=-1)#(1000,64,64,12)
test_a = torch.cat((test_a, gridx.repeat([ntest, 1, 1, 1]), gridy.repeat([ntest, 1, 1, 1])), dim=-1)#(1000,64,64,12)


x_values,y_values=np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,64))
train_u=train_u.numpy()
time_indices=[0,4,9]
for idx in time_indices:
    sample=train_u[499]
    sol=sample[:,:,idx]
    plt.figure()
    plt.pcolormesh(x_values, y_values, sol, shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig(f'ns_1_{idx}.png')
    plt.close()
train_u=torch.tensor(train_u)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
"""
model1 = Net2d(modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')
print("模型的参数总量为：",model1.count_params())
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=scheduler_step, gamma=scheduler_gamma)
myloss1 = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

for ep in range(epochs):
    model1.train()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model1(xx)
            loss += myloss1(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss1(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer1.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer1.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]#y是yy中对应当前时间步的切片，(20,64,64,1)
                im = model1(xx)#输出维度为(20,64,64,1)
                loss += myloss1(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)#（第一次初始化pred为im，之后每次将im拼接到pred上的最后一个维度）

                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)
                #xx[..., step:-2]取出xx最后一个维度上从step到倒数第2个位置的所有值，维度为(20,64,64,9)
                #拼接后xx的维度为(20,64,64,12)，目的是时间推进

            test_l2_step += loss.item()
            test_l2_full += myloss1(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    scheduler1.step()
    print(ep,  train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
# torch.save(model, path_model)
"""
model=torch.load('skip_fno_1e-3_ns.pth')
i=1
r_grid1_x_grid1=torch.zeros_like(train_u)
pred=torch.zeros_like(train_u)
with torch.no_grad():
    for xx, yy in train_loader:
        xx, yy=xx.cuda(),yy.cuda()
        for t in range(0, T, step):
            im = model(xx)
            pred[(i - 1) * batch_size:(i * batch_size),:,:,t:t+1]=im
            r_grid1_x_grid1[(i - 1) * batch_size:(i * batch_size),:,:,t:t+1]=yy[:,:,:,t:t+1]-pred[(i - 1) * batch_size:(i * batch_size),:,:,t:t+1].cuda().detach()
            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]).cuda(), gridy.repeat([batch_size, 1, 1, 1]).cuda()), dim=-1)
        i=i+1

pred0=pred[499,:,:,0].cpu().numpy()
plt.figure()
plt.pcolormesh(x_values, y_values, pred0/10, shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig(f'ns_2_0.png')
plt.close()

pred4=pred[499,:,:,4].cpu().numpy()
plt.figure()
plt.pcolormesh(x_values, y_values, pred4/10, shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig(f'ns_2_4.png')
plt.close()

pred9=pred[499,:,:,9].cpu().numpy()
plt.figure()
plt.pcolormesh(x_values, y_values, pred9/10, shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig(f'ns_2_9.png')
plt.close()

r1=r_grid1_x_grid1[499,:,:,0].cpu().numpy()
plt.figure()
plt.pcolormesh(x_values, y_values, r1, shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig(f'ns_3_0.png')
plt.close()

r4=r_grid1_x_grid1[499,:,:,4].cpu().numpy()
plt.figure()
plt.pcolormesh(x_values, y_values, r4, shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig(f'ns_3_4.png')
plt.close()

r9=r_grid1_x_grid1[499,:,:,9].cpu().numpy()
plt.figure()
plt.pcolormesh(x_values, y_values, r9, shading='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig(f'ns_3_9.png')
plt.close()




# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0;
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})



