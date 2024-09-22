import numpy as np

def stochastic_gradient_descent(X, y, learning_rate, epochs):
    """
    随机梯度下降（SGD）算法实现

    参数:
    X : numpy.ndarray
        特征矩阵，形状为 (m, n)，其中 m 是样本数，n 是特征数
    y : numpy.ndarray
        标签向量，形状为 (m,)
    learning_rate : float
        学习率
    epochs : int
        训练的轮数

    返回:
    weights : numpy.ndarray
        学习后的权重向量
    """
    m, n = X.shape
    weights = np.zeros(n)

    for epoch in range(epochs):
        for i in range(m):
            # 选择一个随机样本
            random_index = np.random.randint(m)
            xi = X[random_index]
            yi = y[random_index]

            # 计算预测值
            prediction = np.dot(xi, weights)

            # 更新权重
            weights -= learning_rate * (prediction - yi) * xi

    return weights

# 示例用法
# 创建一个简单的数据集
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])

# 训练SGD
weights = stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=1000)
print("训练后的权重:", weights)
