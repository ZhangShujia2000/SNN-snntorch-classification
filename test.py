"""
直接运行此程序，可以利用训练好的网络net.pt，可视化测试一张测试集中的图像
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

        # Initialize layers
        self.fc1 = nn.Linear(28*28, 1000)#层与层之间的线性权重
        self.lif1 = snn.Leaky(beta=0.95)#定义隐藏层的脉冲神经元
        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = snn.Leaky(beta=0.95)#定义输出层的脉冲神经元

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(25):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
def show_image(images, labels):
    fig, axes = plt.subplots(1, 4, figsize=(15, 8))
    for index, image in enumerate(images):
        # pytorch中Tensor的shape是[C, H, W],使用matplotlib显示时，需要转换shape到[H, W, C]
        image = image.numpy().transpose(1, 2, 0)
        label = labels[index]
        axes[index].set_title(label)
        axes[index].imshow(image)
def img_show(img):
    img=255*img[0][0]
    print(img)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
data_path='./data/mnist'
net = torch.load('net.pt')
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True, drop_last=False)
with torch.no_grad():
    net.eval()
    data, targets = next(iter(test_loader))
    img_show(data)
    test_spk, test_mem = net(data.view(data.size(0), -1))
    _, predicted = test_spk.sum(dim=0).max(1)
    print("target:",targets.item(),"predicted:",predicted.item() )
