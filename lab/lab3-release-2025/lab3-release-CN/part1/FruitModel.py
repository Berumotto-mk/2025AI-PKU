import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:#空模型
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性
        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数#创建了两个空字典
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()
        self.sigma = 1 # 平滑参数，防止除0错误

    def count(self):# 统计token分布（改变token_num）,要怎样统计呢
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        #正负样本通过label来区分
        for text,label in self.dataset:
            label = int(label)
            #print(text,"count内容",label)
            self.pos_neg_num[label] += 1
            #tokenize_text = text.strip()#strip() 方法用于去除每行文本的首尾空白字符。
            
            #tokenize_text = self.dataset.tokenize(tokenize_text)
            for token in text:
                if token not in self.token_num[0] and token not in self.token_num[1]:
                    self.V +=1
                if token not in self.token_num[label]:
                    self.token_num[label][token] =1
                else:
                    self.token_num[label][token] += 1
        
        
        #raise NotImplementedError # 由于没有返回值，提交时请删掉这一行

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        #tokenize_text = text.strip()
        #print(text)
        #tokenize_text = self.dataset.tokenize(text)
        
        p_p = self.pos_neg_num[1]
        p_n = self.pos_neg_num[0]
        P_p_text = p_p
        P_n_text = p_n
        for token in text:
            #if token not in self.token_num[0] or token not in self.token_num[1]:
                #continue  # 跳过未登录词
            P_p_text *= (self.token_num[1].get(token,0) + self.sigma)/(self.pos_neg_num[1] + self.sigma * self.V)
            P_n_text *= (self.token_num[0].get(token,0) + self.sigma)/(self.pos_neg_num[0] + self.sigma * self.V)
            
        if P_p_text > P_n_text:
            return 1
        else:
            return 0
        raise NotImplementedError


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector#word2vec后每一个token对应的特征向量。每个特征向量大小为100维
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内） 
        ans = np.zeros((max_len,100))#zeros的用法
        index = 0
        for word in text:
            if word  in self.emb:
                ans[index] = self.emb[word]
            index += 1
            if index == max_len:
                break
        return ans

        raise NotImplementedError


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()
    
    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()
        #document_list的结构可以参照fruit
    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        #print(document)
        #print('documnet')
        clean_tokens = document['document']
        word_tf = 0
        for clean_token in clean_tokens:
            if word == clean_token:
                word_tf += 1
        N = len(clean_tokens)
        tf_out = np.log10((word_tf/N)+1)
        return tf_out
            
        
        
        raise NotImplementedError  

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        D = len(self.document_list) #语料库中所有文件数
        d=0#语料库中，该单词出现的frequncy
        for document in self.document_list:
            clean_tokens = document['document']
            if word in clean_tokens:
                d += 1
        idf_out = np.log10(D/(1+d))
        return idf_out
        raise NotImplementedError
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        return self.tf(word,document)*self.idf(word)

        raise NotImplementedError

    def __call__(self, query):
        query = tokenize(query) # 将问题token化,去除了停用词和标点
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
       
        best_tfidf = float('-inf')
        i=0
        best_document = {}
        for document in self.document_list:
            sum_tfidf = 0
            for word in query:
                sum_tfidf += self.tfidf(word,document)
                #print(sum_tfidf,"+")
            if best_tfidf <sum_tfidf:
                #print(sum_tfidf,i,"best_tfidf")
                i += 1
                best_document = document
                best_tfidf = sum_tfidf
        sentences = best_document['sentences']
        
        best_choose = float('-inf')
        best_sentence = ""
        j = 0
        for tokens,sentence in sentences:
            sum_idf = 0
            num = 0
            for word in query:
                if word in tokens:
                    sum_idf += self.idf(word)
                    num +=1
            choose = sum_idf + sum_idf*(num/len(query))
            if choose > best_choose:
                #print(choose,j,"choose")
                j += 1
                best_sentence = sentence
                best_choose = choose
        return best_sentence 
                    
                    
                
            
                
             
        raise NotImplementedError

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 1e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-4  # L2正则化
    batchsize = 64
    max_epoch = 10
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)