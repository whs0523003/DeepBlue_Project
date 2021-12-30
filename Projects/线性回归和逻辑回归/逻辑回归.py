import sys

import numpy as np
import matplotlib.pyplot as plt

# 构建数据X，做一个二分类任务，每条数据2个特征
X = np.array([
             (0.5, 0.5),
             (0.5, 1.0),
             (0.5, 1.5),
             (0.8, 2.0),
             (0.6, 1.2),
             (0.9, 1.3),
             (1.2, 0.9),
             (1.5, 0.5),
             (1.1, 2.9),
             (1.5, 3.0),
             (1.6, 2.5),
             (1.8, 2.0),
             (1.9, 3.1),
             (2.3, 2.8),
             (2.3, 1.6),
             (2.9, 1.5)
             ], np.float32)

# 每条数据对应的标签
label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], "float").T

# 按照索引位置把X的数据分为正类和负类
# label[0:, 0]为行全取，列取第一个。之前label已经通过.T转成一列了
positive = X[label[0:, 0] == 1]
negative = X[label[0:, 0] == 0]

# [:, 0]为行全取，列取第一个，表示每条数据的第一个的特征值
# [:, 1]为行全取，列取第二个，表示每条数据的第二个的特征值
plt.plot(positive[:, 0], positive[:, 1], "go")
plt.plot(negative[:, 0], negative[:, 1], "rx")  # take the x and y of all samples
plt.show()

# 样本数量,16
num_sample = X.shape[0]
# 样本特征数量,2
num_feature = X.shape[1]
# 样本输出数量
num_output = 1
# 因为样本不多，所以在一个batch_size里放所有样本
batch_size = num_sample

# insert a new dim before dim 0  # tip 多画图，图像记忆以后就容易调用了
# 在二维数组X的每个元素的第0个位置插入1，axis=1按列竖着插入
# X_hat大小为(16,3)
X_hat = np.insert(X, 0, values=1, axis=1)  # https://www.tutorialspoint.com/numpy/numpy_insert.htm
# print(X)
# print(X_hat)

# 使用随机正态分布初始化一个值在(0,1)范围内的矩阵，大小为(3,1)
W = np.random.normal(0, 1, size=(1+num_feature, num_output))

lr = 0.1
epochs = 1000

# 定义一个激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # why z is better. mnemonic

# 训练1000个epochs
for i in range(epochs):
    # np.matmul(X_hat, W) 或 np.dot(X_hat, W) 或 X_hat @ W 表示叉乘，即两个矩阵相乘，矢量积
    # np.multiply(X_hat, W) 表示点乘，即两个矩阵中元素一一对应相乘
    # 这里用的是叉乘，(16,3) @ (3,1) = (16,1)
    predict = X_hat @ W
    # 预测结果经过sigmoid激活函数
    probability = sigmoid(predict)
    # 交叉熵损失
    loss = -np.sum(label * np.log(probability) + (1 - label) * np.log(1 - probability)) / batch_size

    del_predict = (probability - label) / batch_size

    # matrix multiplication
    del_W = X_hat.T @ del_predict

    W -= lr * del_W

    # for visualization
    if ((i + 1) % 100 == 0) or (i % 5 == 0 and i / 5 <= 3):
        predict = X_hat @ W
        probability = sigmoid(predict)

        posititve = X[probability[:, 0] >= 0.5]
        negative = X[probability[:, 0] < 0.5]

        bias, theta0, theta1 = W

        tx = np.array([0, 5])
        ty = -(tx * theta0 + bias) / theta1

        plt.title(f"loss: {loss: 3f}")
        plt.plot(positive[:, 0], positive[:, 1], "go")  # plt.plot draws point
        plt.plot(negative[:, 0], negative[:, 1], "rx")
        plt.plot(tx, ty)  # plt.plot also draws line.

        plt.axis([0, 4, 0, 4])
        plt.pause(1)

# 先做更新，更新完后再画图

