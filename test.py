from torchvision import datasets
import torchvision.transforms as transforms
import torch
import numpy as np
#非并行加载就填0
num_workers = 0
#决定每次读取多少图片
batch_size = 20

#转换成张量
transform = transforms.ToTensor()


#下载数据
train_data = datasets.MNIST(root = './drive/data',train = True,\
                           download = True,transform = transform)
test_data = datasets.MNIST(root = './drive/data',train = False,\
                          download = True,transform = transform)

#创建加载器
train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,
                                           num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = batch_size,
                                         num_workers = num_workers)

# 定义MLP模型

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 两个全连接的隐藏层，一个输出层
        # 因为图片是28*28的，需要全部展开，最终我们要输出数字，一共10个数字。
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        hidden_1 = 512
        hidden_2 = 512
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        # 使用dropout防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)
        x = self.fc3(x)
        #     x = F.log_softmax(x,dim = 1)

        return x


model = Net()
# 打印出来看是否正确
print(model)
#定义损失函数和优化器

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(),lr = 0.01)

# 训练
n_epochs = 50

for epoch in range(n_epochs):
    train_loss = 0.0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)  # 得到预测值

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(
        epoch + 1,
        train_loss))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
