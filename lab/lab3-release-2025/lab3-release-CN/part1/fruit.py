from SST_2.dataset import testdataset
import os
import nltk
import string

fruit_dataset = testdataset()


def getdata(index):
    text, label = fruit_dataset.__getitem__(index, show=True)
    return text, label


def tokenize(text):
    cleaned_tokens = []
    tokens = nltk.tokenize.word_tokenize(text.lower())#text.lower全部进行小写,
    #nltk.tokenize.word_tokenize将文本分成单词
    for token in tokens:
        if token in nltk.corpus.stopwords.words('english'):#去除停止符
            continue
        else:
            all_punct = True
            for char in token:
                if char not in string.punctuation:#去除标点符号
                    all_punct = False
                    break
            if not all_punct:
                cleaned_tokens.append(token)
    return cleaned_tokens


def get_document(root='./qadata'):
    cleaned_tokens = []
    document_list = os.listdir(root)#os是操作系统，os.lisdir列出指定目录下的所有文件和子目录
    document_list.sort()
    all_documents = []
    for path in document_list:
        path = os.path.join(root, path)#os.path子模块
        
        with open(path, 'r', encoding='utf-8') as file:
            document = file.read()

        # tokenize document
        cleaned_tokens = tokenize(document)
        now_document = {}
        now_document['document'] = cleaned_tokens

        # tokenize sentences
        sentences = []
        for passage in document.split("\n"):
            for sentence in nltk.sent_tokenize(passage):#用于将一段文本（paragraph）按句子边界分割成单独的句子
                tokens = tokenize(sentence)#一个token的list
                sentences.append([tokens, sentence])

        now_document['sentences'] = sentences
        all_documents.append(now_document)
    return all_documents

