import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd




# 数据预处理
# 将字符串转化成one hot编码形式
def seq2onehot(seq):
    module = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    i = 0
    promoter_onehot = []
    while i < len(seq):
        tmp = []
        for item in seq[i]:
            if item == 't' or item == 'T':
                tmp.append(module[0])
            elif item == 'c' or item == 'C':
                tmp.append(module[1])
            elif item == 'g' or item == 'G':
                tmp.append(module[2])
            elif item == 'a' or item == 'A':
                tmp.append(module[3])
            else:
                tmp.append([0, 0, 0, 0])
        promoter_onehot.append(tmp)
        i = i + 1
    data = np.zeros((len(seq), 118, 1, 4))
    data = np.float32(data)
    m = 0
    while m < len(seq):
        n = 0
        while n < len(seq[0]):
            data[m, n, 0, :] = promoter_onehot[m][n]
            n = n + 1
        m = m + 1
    return data


# random perm the sequence and expression data
def random_perm(seq, exp, shuffle_flag):
    indices = np.arange(seq.shape[0])
    np.random.seed(shuffle_flag)
    np.random.shuffle(indices)
    seq = seq[indices]
    exp = exp[indices]
    return seq, exp


# load data
dataframe = pd.read_table("./seq/source_data_3.tab", sep='\t', header=[1])
dataframe = dataframe.dropna()
dataframe = dataframe.reset_index(drop=True)
promoter = dataframe['Sequence']
expression = dataframe['Core promoter activity ']
data = seq2onehot(promoter)
expression_new = np.zeros((len(expression),))

i = 0
while i < len(expression):
    expression_new[i] = float(expression[i])
    i = i + 1
expression = np.log2(expression_new)
data, expression = random_perm(data, expression, shuffle_flag=3)  # data(11884, 50, 4, 1)
data = data.reshape([5963, 118, 4, 1])
r = 4000
train_feature = data[0:3500]
eval_feature = data[3500:4000]
test_feature = data[r:len(data)]
train_label = expression[0:3500]
eval_label = expression[3500:r]
test_label = expression[r:len(expression)]
np.save('test_feature.npy', test_feature)
np.save('test_label.npy', test_label)
#
#
class train_dataset(Dataset):
    def __init__(self, feature, labels):
        self.feature = feature
        self.labels = labels

    def __getitem__(self, ix):
        self.fea = self.feature[ix][:]
        self.lab = self.labels[ix]
        data = {}
        data['feature'] = torch.from_numpy(np.array(self.fea, dtype=float)).type(torch.FloatTensor)
        data['label'] = torch.from_numpy(np.array(self.lab, dtype=float)).type(torch.FloatTensor)
        return data

    def __len__(self):
        return len(self.labels)

train_dataset = train_dataset(train_feature, train_label)
train_data = DataLoader(train_dataset, batch_size=128)


class eval_dataset(Dataset):
    def __init__(self, feature, labels):
        self.feature = feature
        self.labels = labels

    def __getitem__(self, ix):
        self.fea = self.feature[ix][:]
        self.lab = self.labels[ix]
        data = {}
        data['feature'] = torch.from_numpy(np.array(self.fea, dtype=float)).type(torch.FloatTensor)
        data['label'] = torch.from_numpy(np.array(self.lab, dtype=float)).type(torch.FloatTensor)
        return data

    def __len__(self):
        return len(self.labels)
eval_dataset = eval_dataset(eval_feature, eval_label)
eval_data = DataLoader(eval_dataset, batch_size=128)


class PREDICT(nn.Module):

    # 卷积层结构卷积-ReLU-池化-卷积-ReLU-卷积-ReLU-池化-全连接-ReLU-全连接
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(118, 200, kernel_size=(6, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(200, 400, kernel_size=(5, 1), padding=(3, 0)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(400, 400, kernel_size=(6, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(800, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        x = x.squeeze(-1)
        # x = x.squeeze(-1)
        return x



torch.manual_seed(101)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PREDICT()
model.to(device)
# print(model)
# 设置损失函数
criterion = nn.MSELoss()
criterion = criterion.to(device)
# # 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
# # 网络训练
EPOCHS = 500
Loss = []
Val_loss = []
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    train_steps = 0
    for train_batch, data in enumerate(train_data):
        seq = data['feature'].to(device)
        exp = data['label'].to(device)  # 传到gpu
        # seq = data['feature']
        # exp = data['label']
        optimizer.zero_grad()  # 梯度清零
        train_prediction = model(seq)  # 训练
        batch_train_loss = criterion(train_prediction, exp)  # 计算每个batch的损失
        batch_train_loss.backward()
        optimizer.step()
        epoch_train_loss += batch_train_loss.item()  # 计算epoch的损失
        train_steps += 1
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for eval_batch, data in enumerate(eval_data):
            seq = data['feature'].to(device)
            exp = data['label'].to(device)
            eval_prediction = model(seq)
            eval_loss = criterion(eval_prediction, exp)
            epoch_test_loss += eval_loss.item()
    Loss.append(epoch_train_loss/71)
    Val_loss.append(epoch_test_loss/8)
    np.save('Loss.npy', Loss)
    np.save('Val_loss.npy', Val_loss)
    print(f'Epoch: {epoch + 1:2} Loss: {epoch_train_loss/71} Val_loss: {epoch_test_loss/8}')

torch.save(model.state_dict(), 'CNN_train.pth')
#
#
