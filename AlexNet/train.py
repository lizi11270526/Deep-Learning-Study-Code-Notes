import copy, time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST

from model import AlexNet


# 训练与验证集预处理函数
def train_val_data_process() -> tuple[DataLoader, DataLoader]:
    # 加载 FashionMNIST 训练集，并将图像尺寸调整为 227x227，转换为张量
    train_data = FashionMNIST(root=r"F:\Datasets\FashionMNIST", train=True, transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]), download=True)

    # 按 8:2 比例划分训练集与验证集
    train_data, val_data = random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    # 创建训练集 DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=6)
    # 创建验证集 DataLoade
    val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=True, num_workers=6)
    # 返回训练集和验证集的 DataLoader
    return train_loader, val_loader

# 模型训练函数
def train_model_process(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> pd.DataFrame:
    # 设备类型（CUDA/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数（交叉熵）
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_list = []
    # 验证集损失列表
    val_loss_list = []
    # 训练集准确列表
    train_acc_list = []
    # 验证集准确列表
    val_acc_list = []
    # 当前时间
    since = time.time()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1} / {epochs}')
        print('-'*10)

        # 初始化参数
        # 训练集损失
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_loader):
            # 将特征放到训练设备中
            b_x = b_x.to(device)
            # 将标签放到训练设备中
            b_y = b_y.to(device)
            # 设置模型-训练模式
            model.train()
            # 前向传播，输入一个batch，输出一个batch中对应的预测
            output = model(b_x)
            # 获取每一行中最大值的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)
            # 将梯度置0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据反向传播的梯度信息更新网络参数，降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 若预测正确，train_corrects + 1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 对训练集样本数累加
            train_num += b_x.size(0)
        
        for step, (b_x, b_y) in enumerate(val_loader):
            # 将特征放到验证设备中
            b_x = b_x.to(device)
            # 将标签放到验证设备中
            b_y = b_y.to(device)
            # 设置模型-评估模式
            model.eval()
            # 前向传播，输入一个batch，输出一个batch中对应的预测
            output = model(b_x)
            # 获取每一行中最大值的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)
            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 若预测正确，val_corrects + 1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 对验证集样本数累加
            val_num += b_x.size(0)
        
        # 计算并保存每次迭代的损失值和准确率
        train_loss_list.append(train_loss / train_num)
        train_acc_list.append(train_corrects.double().item() / train_num)
        val_loss_list.append(val_loss / val_num)
        val_acc_list.append(val_corrects.double().item() / val_num)

        print(f'Train Loss: {train_loss_list[-1]}, Train Acc: {train_acc_list[-1]}')
        print(f'Val   Loss: {val_loss_list[-1]}, Val Acc: {val_acc_list[-1]}')

        # 更新最高准确度及权重
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            
        # 训练耗时
        time_use = time.time() - since
        print("Train time: {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    
    # 加载最优模型参数并保存
    torch.save(best_model_wts, r"./best_model.pth")
        
    train_process = pd.DataFrame(data={"epoch": range(epochs), 
                                       "train_loss_list": train_loss_list, 
                                       "train_acc_list": train_acc_list,
                                       "val_loss_list": val_loss_list,
                                       "val_acc_list": val_acc_list})
    return train_process

# 画图
def matplot_acc_loss(train_process: pd.DataFrame):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_list, 'ro-', label = 'train_loss')
    plt.plot(train_process["epoch"], train_process.val_loss_list, 'bs-', label = 'val_loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_list, 'ro-', label = 'train_acc')
    plt.plot(train_process["epoch"], train_process.val_acc_list, 'bs-', label = 'val_acc')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet = AlexNet()

    train_loader, val_loader = train_val_data_process()
    train_process = train_model_process(alexnet, train_loader, val_loader, 20)
    matplot_acc_loss(train_process)