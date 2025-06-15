import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# CNNモデルの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 入力チャンネル数1, 出力チャンネル数32, カーネルサイズ3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 入力チャンネル数32, 出力チャンネル数64, カーネルサイズ3
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)  # 全結合層
        self.fc2 = nn.Linear(128, 10)  # 出力は10クラス（0-9の数字）

    def forward(self, x):
        x = self.conv1(x)  # 畳み込み層1
        x = F.relu(x)  # 活性化関数
        x = self.conv2(x)  # 畳み込み層2
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # プーリング層
        x = self.dropout1(x)  # ドロップアウト1
        x = torch.flatten(x, 1)  # テンソルを平坦化
        x = self.fc1(x)  # 全結合層1
        x = F.relu(x)
        x = self.dropout2(x)  # ドロップアウト2
        x = self.fc2(x)  # 全結合層2
        output = F.log_softmax(x, dim=1)  # 出力層
        return output

# モデルのインスタンス化
model = Net()
print("CNNモデルが作成されました")