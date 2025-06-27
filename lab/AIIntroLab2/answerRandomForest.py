from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 30     # 树的数量
ratio_data = 0.7   # 采样的数据比例
ratio_feat = 0.3 # 采样的特征比例
hyperparams = {
    "depth":5, 
    "purity_bound":1e-2,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    forest = []
    #属性扰动
    for i in range(num_tree):
        # 1.随机采集数据
        n = X.shape[0]
        m = int (n *ratio_data)
        idx = np.random.choice(n,m,replace= False)
        X_sample = X[idx]
        Y_sample = Y[idx]
        # 2.随机采集特征
        #d = X.shape[1]
        #p = int (d*ratio_feat)
        #idy = np.random.choice(d,p,replace=False)
        #X_sample = X_sample[:,idy]
        # 3.构建决策树
        rand_feat = np.random.choice(mnist.num_feat, int(ratio_feat * mnist.num_feat))
        rand_feat=list(rand_feat)
        DecisionTree = buildTree(X_sample,Y_sample,rand_feat,**hyperparams)
        forest.append(DecisionTree)
    return forest
    
    raise NotImplementedError    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
