import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积层：输入通道1(灰度图)，输出通道32，卷积核3x3，步长1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二层卷积层：输入通道32，输出通道64，卷积核3x3，步长1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout层用于防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) # 输出10个类别 (0-9数字)

    def forward(self, x):
        # 卷积 -> 激活 -> 卷积 -> 激活 -> 池化 -> Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # 展平以便输入全连接层
        x = torch.flatten(x, 1)
        
        # 全连接 -> 激活 -> Dropout -> 全连接 -> Softmax
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
