"""
利用此程序可以让dataloader加载自己创建的任意待分类的数据集（不同类别的图像在不同的文件夹下，文件夹名称为类别的名称）
"""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import random
from matplotlib import pyplot as plt


class MyDataset(Dataset):
    # 构造器初始化方法
    def __init__(self, filenames, labels, transforms=None):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms

    # 重写getitem方法用于通过idx获取数据内容
    def __getitem__(self, idx):
        # 使用Pillow Image读取图片文件
        image = Image.open(self.filenames[idx]).convert("RGB")
        # 对图像数据进行转换
        if self.transforms is not None:
            image = self.transforms(image)
        return image, self.labels[idx]

    # 重写len方法获取数据集大小
    def __len__(self):
        return len(self.filenames)


def show_image(images, labels, classes):
    fig, axes = plt.subplots(1, 4, figsize=(15, 8))
    for index, image in enumerate(images):
        # pytorch中Tensor的shape是[C, H, W],使用matplotlib显示时，需要转换shape到[H, W, C]
        image = image.numpy().transpose(1, 2, 0)
        label = labels[index]
        axes[index].set_title(classes[label])
        axes[index].imshow(image)


# 定义图像预处理转换方法
transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms处理的目标是Image对象
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomGrayscale(p=0.3),
        # 将Image对象转换为Tensor张量
        torchvision.transforms.ToTensor()
    ]
)

image_dataset = torchvision.datasets.ImageFolder("")#这里是数据集的地址
# image_dataset.samples 中存放的是图像数据的文件路径和类别索引编号（从0开始编号）
random.shuffle(image_dataset.samples)
# image_dataset.classes 列表中存放的类别顺序与image_dataset.samples中存放的类别索引编号相对应
classes = image_dataset.classes
# print(image_dataset.samples[:5])

# 用于存放图像路径列表
filenames = []
# 用于存放图像对应的类别
labels = []
for image_path, label in image_dataset.samples:
    #     print(image_path, label)
    filenames.append(image_path)
    labels.append(label)

dataset = MyDataset(filenames, labels, transforms)
dataloader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True, drop_last=False)

for images, labels in dataloader:
    print(images.shape, labels)
    # 显示读取到的图像数据，并验证类别信息是否真确
    show_image(images, labels, classes)