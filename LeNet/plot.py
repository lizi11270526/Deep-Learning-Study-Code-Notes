import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


train_data = FashionMNIST(root=r"F:\Datasets\FashionMNIST", train=True, transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
    
    # 四维张量移除第一维，转为Numpy数组
    batch_x = b_x.squeeze().numpy()
    # 张量转为Numpy数组
    batch_y = b_y.numpy()
    # 训练集的标签
    class_label = train_data.classes
    print(f"class_label: {class_label}")

    # 创建画布，设置图像尺寸（宽14英寸，高5英寸）
    plt.figure(figsize=(14, 5))
    for i in np.arange(len(batch_y)):
        # 在4行16列的网格中，定位第 i+1 个子图
        plt.subplot(4, 16, i + 1)
        # 显示灰度图像（batch_x 形状为 [batch, 224, 224]）
        plt.imshow(batch_x[i, :, :], cmap='gray')
        # 设置子图标题为对应的类别名称
        plt.title(class_label[batch_y[i]], size=10)
        # 隐藏坐标轴
        plt.axis('off')
        # 调整子图之间的水平间距
        plt.subplots_adjust(wspace=0.05)
    # 渲染显示图像
    plt.show()
