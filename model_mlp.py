from load_data import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, img_size, class_num):
        super(mlp, self).__init__()
        self.img_size = img_size
        hidden_1 = img_size**2
        hidden_2 = img_size**2
        hidden_3 = img_size**2
        self.fc1 = nn.Linear(img_size**2*3, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_2, hidden_3)
        self.fc5 = nn.Linear(hidden_3, class_num)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, img_size**2*3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        x = F.softmax(x,dim=1)
        return x


class_num = 3



class_list = '[[4], [2], [1]]'
# class_list = '[[1], [2], [3], [4], [5], [6], [7], [14], [16], [17], [20], [21], [22], [23], [24]]'
class_num = len(eval(class_list))
img_size = 50

train_X, train_Y, test_X, test_Y  = load_data(data_path = r'./data/cutted_data', size = img_size, class_list = class_list)
train_Y = np.expand_dims(train_Y, axis=1)
test_Y = np.expand_dims(test_Y, axis=1)


train_X, test_X = \
    train_X * 1.0 / 127.5 - 1,\
    test_X * 1.0 / 127.5 - 1
train_X, test_X = \
    train_X.reshape((train_X.shape[0],train_X.shape[1]**2*3)),\
    test_X.reshape((test_X.shape[0],test_X.shape[1]**2*3))




train_X, test_X, train_Y, test_Y = \
    torch.from_numpy(train_X).float(),\
    torch.from_numpy(test_X).float(),\
    torch.from_numpy(train_Y).long(),\
    torch.from_numpy(test_Y).long()


model = mlp(img_size, class_num)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(),lr = 0.01)

print('training..')
n_epochs = 50
pre_loss = 10000
loss_threshold = .1
for epoch in range(n_epochs):
    train_loss = 0.0

    for i in range(len(train_X)):
        data, target = train_X[i], train_Y[i]
        optimizer.zero_grad()
        output = model(data)  # 得到预测值
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
    if pre_loss - train_loss < loss_threshold:
        break
    pre_loss = train_loss
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(
        epoch + 1,
        train_loss))


#测试
# initialize lists to monitor test loss and accuracy

model.eval() # prep model for *evaluation*

correct_count = 0
correct_count_list = [0]*class_num
class_count = [0]*class_num
for i in range(len(test_X)):
    data = test_X[i]
    target = test_Y[i]
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    lable = int(output.argmax())
    class_count[target] += 1
    if lable == target:
        correct_count += 1
        correct_count_list[target] += 1

print("total_accuracy: %.2f%%"%(100*correct_count/len(test_X)))
for i in range(len(class_count)):
    print("class%d_accuracy: %.2f%%"%(i, 100*correct_count_list[i]/class_count[i]))