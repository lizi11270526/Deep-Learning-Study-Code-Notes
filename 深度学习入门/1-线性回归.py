# 数据特征
x_data = [1, 2, 3]  
# 数据标签
y_data = [2, 4, 6]
# 权重
w = 4
# 学习率
learning_rate = 0.01

# 线性回归模型
def forward(x):
    return w * x

# 损失函数
def cost(xs, ys):
    cost_value = 0
    for x, y in zip(xs, ys):
        # 预测值
        y_pred = forward(x)
        # 计算损失
        cost_value += (y_pred - y) ** 2
    return cost_value / len(xs)

# 梯度下降函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (w * x - y)
    return grad / len(xs)

# 训练模型
for epoch in range(100):
    # 计算损失
    cost_value = cost(x_data, y_data)
    # 计算梯度
    grad = gradient(x_data, y_data)
    # 更新权重
    w -= learning_rate * grad
    # 打印损失和权重
    print(f'Epoch: {epoch}, Weight: {w}, Loss: {cost_value}')

# 推理模型
print(f'预测值: {forward(4)}')
