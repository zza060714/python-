import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
import os

def train():
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 实例化模型并移动到设备上
    model = Net().to(device)
    
    # 定义优化器 (Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 数据预处理：转换为张量并进行标准化 (MNIST数据集的均值和标准差)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载MNIST训练集
    print("Loading MNIST dataset...")
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    epochs = 3  # 训练轮数
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            # 使用负对数似然损失函数
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # 打印训练进度
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

    # 保存训练好的模型权重
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved to mnist_cnn.pth")

if __name__ == '__main__':
    train()
