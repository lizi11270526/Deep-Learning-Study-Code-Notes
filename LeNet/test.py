import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from model import LeNet


# 测试集预处理函数
def test_data_process() -> tuple[DataLoader, list]:
    # 加载 FashionMNIST 测试集，并将图像尺寸调整为 28x28，转换为张量
    test_data = FashionMNIST(root=r"F:\Datasets\FashionMNIST", train=False, transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]), download=True)
    # 获取标签列表
    classes = test_data.classes
    # 创建测试集 DataLoader
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0)
    return test_loader, classes

# 模型测试函数
def test_model_process(model: nn.Module, test_loader: DataLoader, classes: list) -> None:
    # 设备类型（CUDA/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型放入训练设备中
    model = model.to(device)
    # 初始化参数
    # 测试集精度
    test_corrects = 0.0
    # 测试集样本数量
    test_num = 0

    # 只进行前向传播，不进行梯度计算
    with torch.no_grad():
        for b_x, b_y in test_loader:
            # 将特征放到训练设备中
            b_x = b_x.to(device)
            # 将标签放到训练设备中
            b_y = b_y.to(device)
            # 设置模型-评估模式
            model.eval()
            # 前向传播，输入一个batch，输出一个batch中对应的预测
            output = model(b_x)
            # 获取每一行中最大值的行标
            pre_lab = torch.argmax(output, dim=1)
            # 若预测正确，test_corrects + 1
            test_corrects += torch.sum(pre_lab == b_y.data)
            # 对测试集样本数累加
            test_num += b_x.size(0)
            # 分类结果下标
            result = pre_lab.item()
            # 真实结果下表
            label = b_y.item()
            # 打印分类结果和真实标签
            print(f"Result: {classes[result]} ------Label: {classes[label]}")
        # 计算测试集准确率
        test_acc = test_corrects.double().item() / test_num
        print(f"Test Acc: {test_acc}")
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    model.load_state_dict(torch.load(r"./best_model.pth"))

    test_loader, classes = test_data_process()
    test_process = test_model_process(model, test_loader, classes)