import numpy as np
from PIL import Image, ImageFilter 
# PTL库
#读取数据
#trn_X, trn_Y # 训练集的图片与标签。每个标签是一个0-9的整数。
trn_X = np.load('MNIST/train_data.npy').astype(np.float64)
#'MNIST/train_data.npy'一个二进制文件，导入后生成一个numpy数组
trn_Y = np.load('MNIST/train_targets.npy')
trn_num_sample = trn_X.shape[0]#图片个数
trn_X = trn_X.reshape(trn_num_sample, -1) #将每张图片展平为一个784维的向量
std_X, mean_X = np.std(trn_X, axis=0, keepdims=True)+1e-4, np.mean(trn_X, axis=0, keepdims=True)
#.mean() 方法用于计算数组的均值（平均值）。均值是数据集中所有元素的总和除以元素的数量，是衡量数据集中趋势的一个重要指标。
#.std() 方法用于计算数组的标准差，每个元素与均值的偏差的平方的平均值的平方根
#keepdims = True表示保持原有的维度
# 按列计算均值（axis=0）
num_feat = trn_X.shape[1]#784
#num_feat = 784 # 每张图片是28*28的矩阵。在训练集和验证集中，图片已经展平成一个784维的向量。在测试集中没有展平。
num_class = np.max(trn_Y) + 1 #1-10的数字？

#val_X, val_Y # 验证集的图片与标签。
val_X = np.load('MNIST/valid_data.npy').astype(np.float64)
val_Y = np.load('MNIST/valid_targets.npy')
val_X = val_X.reshape(val_X.shape[0], -1)

test_X = np.load('MNIST/test_data.npy').astype(np.float64)
test_Y = np.load('MNIST/test_targets.npy')

num_data = test_X.shape[0]

def getdata(idx):
    return test_X[idx]

def gety(idx):
    return test_Y[idx]

def getdatasets(y):
    return np.arange(num_data)[test_Y==y]
