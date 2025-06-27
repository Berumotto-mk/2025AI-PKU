import numpy as np

# 超参数



# TODO: You can change the hyperparameters here

lr = 3  # 学习率
wd = 1e-3  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """

    # TODO: YOUR CODE HERE

    haty = np.dot(X,weight)+bias
    return haty
    raise NotImplementedError

def sigmoid(x):
    if x<-(1e+2):
        return np.exp(x) / (np.exp(x) + 1)
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """

    # TODO: YOUR CODE HERE

    haty = np.dot(X,weight)+bias #haty:(n,)
    haty_y = haty*Y
    loss = 0
    for item in haty_y:
        if item <-(1e+2):
            loss += (((-item)+np.log(1+np.exp(item))) + wd * np.dot(weight, weight))
        else:
            loss += (-1)*(np.log(sigmoid(item)))+ wd*np.dot(weight,weight)
    loss = loss / (Y).shape[0]
    
    #loss = np.mean(np.log(1+ np.exp(Y*haty)) + 1e-10) + wd * np.sum(weight**2)
    #weight_grad = -np.mean((1-sigmoid_haty))*np.dot(Y,X) + 2 * wd * weight
    #bias_grad = -np.mean((1-sigmoid_haty)*Y)
    weight_grad = np.zeros(weight.shape)
    bias_grad = 0
    i = 0
    for item in haty_y:
        weight_grad += -(1-sigmoid(item))*Y[i]*X[i]
        bias_grad += -(1-sigmoid(item))*Y[i]
        i +=1 
    weight -= lr * (weight_grad )/(Y).shape[0]+ 2 * wd * weight
    bias -= lr * (bias_grad)/(Y).shape[0]
    return haty,loss,weight,bias
    raise NotImplementedError