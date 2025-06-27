import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {
    "depth": 6, 
    "purity_bound": 1, 
    "gainfunc": "negginiDA" 
    }

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: (n,), 标签向量
    @return: 熵
    """
    # TODO: YOUR CODE HERE
    en = 0
    num = Y.shape[0]
    k,num_k = np.unique(Y,return_counts=True)
    for i in range(len(k)):
        p = num_k[i] /num
        en -= p * np.log2(p)
    return en
    raise NotImplementedError


def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]#取出第idx个特征
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    # TODO: YOUR CODE HERE
    for i in range(len(ufeat)):
        ret -= featp[i] * entropy(Y[feat == ufeat[i]]+EPS)
    ret += entropy(Y+EPS)
    
    return ret
    raise NotImplementedError

def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: (n,), 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点列表，其中key是特征的取值，value是子节点（Node）
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return: bool, 是否为叶节点
        """
        return len(self.children) == 0


def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    """
    递归构建决策树。
    @params X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @params Y: (n,), 样本的label
    @params unused: List of int, 未使用的特征索引
    @params depth: int, 树的当前深度
    @params purity_bound: float, 熵的阈值
    @params gainfunc: Callable, 信息增益函数
    @params prefixstr: str, 用于打印决策树结构
    @return: Node, 决策树的根节点
    """
    
    root = Node()
    u, ucnt = np.unique(Y, return_counts=True)
    root.label = u[np.argmax(ucnt)]
    # print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    # 当达到终止条件时，返回叶节点
    # TODO: YOUR CODE HERE
    if depth == 0 or entropy(Y) < hyperparams["purity_bound"]:
        return root
    if len(unused) == 0:
        return root
    #raise NotImplementedError
    gains = [gainfunc(X, Y, i) for i in unused]
    idx = np.argmax(gains)
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)
    feat = X[:, root.featidx]
    ufeat = np.unique(feat)
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    # TODO: YOUR CODE HERE
    for leaf in ufeat:
        #mask = [feat == ufeat[i]]
        sonY=np.array([Y[i] for i in range(Y.shape[0]) if feat[i]==leaf])
        #print(Y.shape)
        #print(entropy(Y))
        sonX=np.array([X[i] for i in range(Y.shape[0]) if feat[i]==leaf])
        child = buildTree(sonX,sonY,unused,depth+1,purity_bound,gainfunc,prefixstr + "")
        root.children[leaf] = child
    #if depth >= hyperparams["depth"] or entropy(Y) < hyperparams["purity_bound"]:
    #raise NotImplementedError
    return root


def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

