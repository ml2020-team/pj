import torchvision.models as models;
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.autograd import Variable

model=models.resnet18(pretrained=True)
#加载数据
x_train = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/x_train.npy")
y_train = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/y_train.npy")
x_test = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/x_test.npy")
y_test = np.load("E:/2020-2021/机器学习理论/MLdata/task1/size=64/1_task1_trgt_padding/y_test.npy")
x_train = torch.from_numpy(x_train)
x_train=np.transpose(x_train,(0,3,1,2))
x_train = x_train.type('torch.FloatTensor')
y_train = torch.from_numpy(y_train)
y_train = y_train.type('torch.LongTensor')
x_test = torch.from_numpy(x_test)
x_test=np.transpose(x_test,(0,3,1,2))
x_test = x_test.type('torch.FloatTensor')
y_test = torch.from_numpy(y_test)
y_test = y_test.type('torch.LongTensor')
train_dataset=TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataset=TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

#超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH=20
pre_epoch=0
LR=0.0001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 ==0:
            print('Train Epoch: {}\tLoss: {:.6f};'.format(epoch, loss.item()))

def test():
    test_loss = 0
    correct = 0
    # i = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        # item()方法把tensor转为数字
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.item(), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(EPOCH):
    train(epoch)
    test()