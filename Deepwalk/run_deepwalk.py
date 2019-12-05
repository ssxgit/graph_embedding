#!-*- coding:utf8-*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random_walk import deepwalk
path = '../data/cora/cora.cites'
G = nx.DiGraph()

#读取文件构建图
with open(path,'r') as f:
	k=1
	for line in f:

		cols = line.strip().split()

		G.add_weighted_edges_from([(cols[0], cols[1], 1)]) #初始权重为1
		G.add_weighted_edges_from([(cols[1], cols[0], 1)])
		#print(k)
		k+=1

#图g 每个结点随机游走次数 每个结点随机游走长度 最终生成的向量长度
deepwalk = deepwalk(G,10,20,size=100) #deepwalk算法生成向量
deepwalk.save_embeddings("output_deepwalk.weight") #向量存储