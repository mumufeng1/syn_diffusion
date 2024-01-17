import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr


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
model = PREDICT()
model.load_state_dict(torch.load('CNN_train.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)
test_feature = np.load('test_feature.npy')
test_label = np.load('test_label.npy')


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

test_dataset = train_dataset(test_feature, test_label)
test_data = DataLoader(test_dataset)
model.eval()
pred = []
for data in test_data:
    seq = data['feature'].to(device)
    exp_pre = model(seq)
    exp_pre = exp_pre.cpu()
    pred.append(exp_pre.detach().numpy())

# for seq in test_feature:
#     seq = torch.tensor(seq)
#     seq = seq.unsqueeze(0)
#     seq = seq.to(device)
#     exp_pre = model(seq).to(torch.float32)
#     exp_pre = exp_pre.cpu()
#     pred.append(exp_pre.detach().numpy())

# print(pred)
pred = np.array(pred).squeeze()
test_label = np.float32(test_label)
print(pred)
# print(pred.shape)
# print(test_label.shape)
cor_pearsonor = pearsonr(test_label, pred)
print(cor_pearsonor)

# 结果可视化
plt.figure(figsize=(6, 4))
plt.scatter(test_label, pred)
plt.xlabel('test_label')
plt.ylabel('pred')
plt.show()


