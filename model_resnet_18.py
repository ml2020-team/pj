import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.autograd import Variable
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=3):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #print('forward')
        x = np.transpose(x,(0,3,1,2))
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)

#加载数据
x_train = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/x_train.npy")
y_train = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/y_train.npy")
x_test = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/x_test.npy")
y_test = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/y_test.npy")
x_train = torch.from_numpy(x_train)
x_train = x_train.type('torch.FloatTensor')
#print(x_train.shape)
y_train = torch.from_numpy(y_train)
y_train = y_train.type('torch.LongTensor')
x_test = torch.from_numpy(x_test)
x_test = x_test.type('torch.FloatTensor')
y_test = torch.from_numpy(y_test)
y_test = y_test.type('torch.LongTensor')
train_dataset=TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset=TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH=10
pre_epoch=0
LR=0.0001

#模型定义
net = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-3)
x=[]
y=[]
plt.figure(figsize=(10,10))
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        #print('train')
        data, target = Variable(data), Variable(target)
        #print(data.shape)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 ==0:
            print('Train Epoch: {}\tLoss: {:.6f};'.format(epoch, loss.item()))
            x.append(epoch)
            y.append(loss.item())


def test():
    test_loss = 0
    correct = 0
    # i = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        print(data.shape)
        output = net(data)
        loss = criterion(output, target)
        test_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        # item()方法把tensor转为数字
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        # i = i +1

    # print(i)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.item(), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(EPOCH):
    train(epoch)
    test()
    plt.plot(x, y, color='navy')
    plt.title('loss')
    plt.show()







