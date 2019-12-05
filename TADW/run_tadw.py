# -*- coding: utf-8 -*-
import numpy as np
from sentence2vec import word2vec_sentence2vec
from TADW import TADW

node_path = '../data/cora/cora.content'
edge_path = '../data/cora/cora.cites'

# # 计算邻接矩阵A
ids, labels = [], []
with open(node_path, 'r') as f:
    line = f.readline()
    while line:
        line_split = line.split()

        ids.append(line_split[0])
        labels.append(line_split[-1])
        line = f.readline()
ID = dict()
index = 0
for x in ids:
    ID[x] = index
    index += 1

A = np.zeros((len(ids), len(ids)))
with open(edge_path, 'r') as f:
    line = f.readline()
    while line:
        line_split = line.split()
        A[ID[line_split[0]]][ID[line_split[1]]] = 1

        line = f.readline()
## 结点文本矩阵T
T = word2vec_sentence2vec(size=30)

#邻接矩阵 文本矩阵 最终生成的向量长度
tadw = TADW(A,T,100)
tadw.save_embeddings("output_TADW.weight")  #保存结点向量