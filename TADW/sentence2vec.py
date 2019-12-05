# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la
from gensim.models import word2vec

### 结点向量转化
### 两个方案
### 1、word2vec
### 2、svd分解
def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = idx_features_labels[:,1:-1]
    return features
def word2vec_sentence2vec(size=100):
    features=load_data()
    features = features.astype(np.int32)
    sentences = [list(map(str, np.where(feature == 1)[0])) for feature in features]

    model = word2vec.Word2Vec(sentences, window=15,size=size, min_count=0, sg=1)

    sentences_array = []
    for sentence in sentences:
        word_num = len(sentence)
        vectors = []
        for i in range(word_num):
            vectors.append(model.wv[sentence[i]])
        sentence_vector = np.sum(np.array(vectors),axis=0)
        sentences_array.append(sentence_vector)
    return np.array(sentences_array)
#print(word2vec_sentence2vec(100).shape)

def svd(size=100):
    features=load_data()
    X = features.astype(np.int32)
    U, sigma, VT = la.svd(X)
    T = np.dot(U[:,:size],np.diag(sigma[:size]))  #取最大的几个特征值与特征向量相乘
    return T
# print(svd().shape)
# print(word2vec_sentence2vec().shape)
