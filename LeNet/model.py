import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层（输入通道1，输出通道6，核大小5，填充2） -> 输出（）
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 激活函数
        self.sig = nn.Sigmoid()
        # 平均池化层（核大小2，步幅2）
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层（输入通道6，输出通道16，核大小5，填充0）
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        # 平均池化层（核大小2，步幅2）
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 平坦层
        self.flatten = nn.Flatten()
        # 全连接层
        self.f5 = nn.Linear(in_features=400, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        # 卷积
        x = self.sig(self.c1(x))
        # 池化
        x = self.s2(x)
        # 卷积
        x = self.sig(self.c3(x))
        # 池化
        x = self.s4(x)
        # 平坦
        x = self.flatten(x)
        # 全连接
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x         
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))