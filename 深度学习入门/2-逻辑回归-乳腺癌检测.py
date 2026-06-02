import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# 读取数据
dataset_path = r"F:\Datasets\1-逻辑回归\breast_cancer_data.csv"
dataset = pd.read_csv(dataset_path)

# 提取X(特征)
X = dataset.iloc[:,:-1]

# 提取Y(标签)
Y = dataset.iloc[:,-1]

# 划分数据集，测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# 进行数据预处理(归一化)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train, y_train)

# 打印模型参数
print(f"w:{lr.coef_}\nb:{lr.intercept_}")

# 推理模型
pre_result = lr.predict(x_test)
print(f"pre_result:{pre_result}")

# 打印预测结果的概率
pre_result_proba = lr.predict_proba(x_test)
print(f"pre_result_proba:{pre_result_proba}")

# 获取第二列概率(恶性肿瘤的概率)
pre_list = pre_result_proba[:,1]
print(f"pre_list:{pre_list}")

# 设定阈值
thresholds = 0.3

# 预测结果列表
result = []
result_name = []
for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append("恶性")
    else:
        result.append(0)
        result_name.append("良性")

print(f"result:{result}")
print(f"result_name:{result_name}")

# 输出结果的精确率、召回率，F1分数
report = classification_report(y_test, result, labels=[0, 1], target_names=["良性", "恶性"])
print(f"report:{report}")

