"""
SNN For Mnist Dataset Classification
author:Zhang Shujia
直接运行此程序，可以下载Mnist数据集并训练。
"""
import snntorch as snn
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

#定义网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3层全连接网络，类似于3层BP算法，但是隐藏层节点是脉冲神经元
        num_inputs = 28 * 28
        num_hidden = 500# 可调
        num_outputs = 10
        # 定义单批数据要经过几个时间步
        beta = 0.95# 衰减率，网络结构的超参数，可调
        # 初始化层
        self.fc1 = nn.Linear(num_inputs, num_hidden)#层与层之间的线性权重
        self.lif1 = snn.Leaky(beta=beta)#定义隐藏层的脉冲神经元
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)#定义输出层的脉冲神经元

    def forward(self, x):

        # 初始化隐藏层结点在t=0时的状态
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # 记录输出层的输出
        spk2_rec = []
        mem2_rec = []
        num_steps = 25 # 时间步 time step
        for step in range(num_steps):
            cur1 = self.fc1(x)#线性变换层
            spk1, mem1 = self.lif1(cur1, mem1) #LIF神经元
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))#output是网络输出的尖峰的集合
    _, idx = output.sum(dim=0).max(1)#找出输出尖峰最多的那个类别结点下标idx并和标签比较，相等则说明分类正确。
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}")
    print(f"Train Set Loss for a single minibatch: {loss_hist[epoch]:.2f}")
    print(f"Test Set Loss for a single minibatch: {test_loss_hist[epoch]:.2f}")


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == '__main__':
    #加载图像并图像预处理
    batch_size = 128
    data_path='./data/mnist'

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 图像预处理
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    #定义超参数
    num_epochs = 5
    lr=1e-3
    num_steps = 25  # 时间步 time step
    #加载网络
    net = Net().to(device)
    #定义优化器和损失函数
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #加载空列表以存储loss值
    loss_hist = []
    test_loss_hist = []
    acc_hist=[]
    test_acc_hist=[]
    #初始化训练参数
    epoch = 0

    # 训练前计算初始准确率
    total = 0
    correct = 0
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)  # 要全部的数据
    with torch.no_grad():
        net.eval()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            train_spk, train_mem = net(data.view(data.size(0), -1))
            _, predicted = train_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        train_acc = torch.zeros((1), dtype=dtype, device=device)
        train_acc += correct / total
        acc_hist.append(train_acc.item())

    total = 0
    correct = 0
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            test_spk, test_mem = net(data.view(data.size(0), -1))
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        test_acc = torch.zeros((1), dtype=dtype, device=device)
        test_acc += correct / total
        test_acc_hist.append(test_acc.item())


    #开始训练
    while epoch < num_epochs:
        # 加载用于训练的数据集
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
        counter = 0
        train_batch = iter(train_loader)
        #批量更新
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # 前向传播
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            # 初始化损失
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)#计算单批数据的所有时间步的总损失

            # 计算梯度并用对应优化算法更新权重
            optimizer.zero_grad()
            loss_val.backward()#损失回传，计算梯度
            optimizer.step()#使用对应的优化器算法更新权重

            # 记录训练前的初始损失
            if counter==0 and epoch==0:
                loss_hist.append(loss_val.item())

            # 验证集，只计算损失，不参与梯度回传
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                #前向传播
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                #计算损失
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                if counter == 0 and epoch==0:
                    test_loss_hist.append(test_loss.item())
                    train_printer()
                    print(f"Train set accuracy for whole dataset: {train_acc.item() * 100:.2f}%")
                    print(f"Test set accuracy for whole dataset: {test_acc.item() * 100:.2f}%")
                    print('\n')

            counter += 1#每一个epoch训练到了第几批
        epoch=epoch+1#训练到了几代
        #每一代训练结束后，记录一下训练集和验证集的单批损失值
        loss_hist.append(loss_val.item())
        test_loss_hist.append(test_loss.item())
        train_printer()

        # 每代训练结束后计算一下训练集和验证集的总准确率
        total = 0
        correct = 0
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)#要全部的数据
        with torch.no_grad():
            net.eval()
            for data, targets in train_loader:
                data = data.to(device)
                targets = targets.to(device)
                train_spk, train_mem = net(data.view(data.size(0), -1))
                _, predicted = train_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            train_acc = torch.zeros((1), dtype=dtype, device=device)
            train_acc += correct/total
            print(f"Train set accuracy for whole dataset: {train_acc.item() * 100:.2f}%")
            acc_hist.append(train_acc.item())

        total = 0
        correct = 0
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
        with torch.no_grad():
            net.eval()
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)
                test_spk,  test_mem= net(data.view(data.size(0), -1))
                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            test_acc = torch.zeros((1), dtype=dtype, device=device)
            test_acc += correct/total
            print(f"Test set accuracy for whole dataset: {test_acc.item() * 100:.2f}%")
            test_acc_hist.append(test_acc.item())

            print("\n")

    # 保存
    torch.save(net,'net.pt')

    # Loss
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    print(loss_hist)
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Accuracy
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    print(acc_hist)
    plt.plot(acc_hist)
    plt.plot(test_acc_hist)
    plt.title("Accuracy Curves")
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()



