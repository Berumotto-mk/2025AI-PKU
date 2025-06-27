import os
import random
import string
import nltk
import pickle

class basedataset():
    def __init__(self, mode, shuffle=False, maxlen=None):#mode是
        assert mode in ['train', 'test', 'dev']#assert用来检查mode是否合法
        self.root = './SST_2/'+mode+'.tsv'#文件路径
        f = open(self.root, 'r', encoding='utf-8')#使用 open 函数以只读模式（'r'）打开文件，并指定文件编码为 UTF-8。
        L = f.readlines()#使用 readlines() 方法读取文件的所有行，并将它们存储在列表 L 中。
        self.data = [x.strip().split('\t') for x in L]#strip() 方法用于去除每行文本的首尾空白字符。
        #,split('\t')：将去除空白字符后的字符串 x 按照制表符 \t 分割成多个字段（相当返回的是句子），返回一个列表。
        if maxlen is not None:
            self.data = self.data[:maxlen]#把data的长度限制在maxlen
        self.len = len(self.data)
        self.D = []
        for i in range(self.len):
            self.D.append(i)
        if shuffle:
            random.shuffle(self.D)
        self.count = 0

    def tokenize(self, text):
        cleaned_tokens = []
        tokens = nltk.tokenize.word_tokenize(text.lower())#将输入的文本字符串 text 转换为小写形式
        #使用nltk.tokenize.word_tokenize()函数将文本分割成单词（tokens）。
        #nltk是自然语言处理的一个库，word_tokenize是nltk中的一个函数，用于将文本分割成单词。
        for token in tokens:
            if token in nltk.corpus.stopwords.words('english'):#过滤掉停用词
                continue
            else:
                all_punct = True
                for char in token:
                    if char not in string.punctuation:#去除标点符号
                        #string.punctuation是一个包含所有标点符号的字符串
                        all_punct = False
                        break
                if not all_punct:
                    cleaned_tokens.append(token)
        return cleaned_tokens

    def __getitem__(self, index, show=False):
        index = self.D[index]
        text, label = self.data[index]
        tokenize_text = text.strip()#strip() 方法用于去除每行文本的首尾空白字符。
        tokenize_text = self.tokenize(tokenize_text)
        if show == True:
            return (tokenize_text, text), int(label)
        else:
            return tokenize_text, int(label)
    
    def get(self, index):
        index = self.D[index]
        text, label = self.data[index]
        return text, int(label)

def traindataset(shuffle=False):
    return basedataset('train', shuffle)

def minitraindataset(shuffle=False):
    return basedataset('train', shuffle, maxlen=128)

def testdataset(shuffle=False):
    return basedataset('dev', shuffle=False)

def validationdataset(shuffle=False):
    return basedataset('dev', shuffle=False)

if __name__ == '__main__':
    pass