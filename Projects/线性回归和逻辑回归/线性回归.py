import sys
import numpy as np
import matplotlib.pyplot as plt
import random

# 我们定义房价数据
# x指年份，减去了2009
# 10个样本，1个特征
# 10 x 1
# kp: [1,2] and [[1,2]] 样本数和特征数
x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.float32).T
print(x.shape)

# y指房价，除以了10000
# 10个样本，每个样本有一个真值
# 10 x 1
# 这里习惯用列向量，一行表示一个样本
y = np.array([[1.8, 2.1, 2.3, 2.3, 2.85, 3.0, 3.3, 4.9, 5.45, 5.0]], dtype=np.float32).T
print(y.shape)

# 之后我们一般用normalization kp: 归一化
x = x / 10.0
y = y / 10.0


# 训练
lr = 0.1
k = random.random()
b = 0
epochs = 500
batch_size = x.shape[0]


# 创建一个新的画板
plt.figure()

# 训练500个epochs
for i in range(epochs):
    # 1. 做预测
    predict = k * x + b

    # 2. 根据预测结果计算loss
    # 使用MSE计算loss
    # 把一个batch的数据的loss相加再除以batch_size,求得该batch数据的平均loss
    loss = np.sum((y - predict) ** 2) * 0.5 / batch_size  # 有奖竞猜为什么要除以batch_size

    # 3. 计算更新量
    # G
    # 写法1 分步写
    del_g = (y - predict) / batch_size  # 有奖竞猜 既然 loss都已经除以了batchsize，这里为何又不需要了呢，求和符号去哪里了？
    del_predict = del_g * (-1)
    # W
    del_k = np.sum(del_predict * x)  # 有奖竞猜，这里为什么不用了/batchsize
    del_b = np.sum(del_predict * 1)

    # 写法2 一步写出来
    #     del_k = np.sum((predict - y)*x)
    #     del_b = np.sum(predict - y)

    # 4. 去更新
    k = k - lr * del_k
    b = b - lr * del_b

    # 5.（optional）log  ----> 以下的内容属于常用工具库的使用
    if (i + 1) % 100 == 0 or i < 3:
        # print(f"{i}") # for debug
        # 为了画模型的直线显示而用
        tx = np.array([[0, 1]]).T  # tx ty 用来画两个点，两点一线
        ty = k * tx + b
        # print(tx, ty, k, b)

        # plt.close()关闭画布
        # plt.clf()清空画布
        plt.clf()  # 清理掉figure ref: https://stackoverflow.com/questions/16661790/difference-between-plt-close-and-plt-clf
        plt.title(f"Iter: {i}, Loss:{loss:.3f}, k: {k:.3f}, b: {b:.3f}")
        plt.plot(x, y, "r.")  # kp: matplot usage    ref : https://github.com/matplotlib/cheatsheets   or   https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf
        plt.plot(tx, ty, "g-")

        # 锁定坐标系
        plt.axis([0, 1, 0, 1])  # 尝试去改一改，搞一搞

        # 在窗口界面下有效，jupyter下无效
        # 等待并刷新，时间是1秒
        plt.pause(1)

