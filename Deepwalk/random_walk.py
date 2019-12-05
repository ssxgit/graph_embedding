#!-*- coding:utf8-*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gensim

class deepwalk(object):
	def __init__(self,g,num,deep_num,size):
		self.g = g
		self.num = num #每个结点随机游走次数
		self.deep_num = deep_num #每个结点随机游走长度
		self.size = size #生成的向量长度
		self.prduce_sentence()
		self.word2vec()

    #保存向量
	def save_embeddings(self, filename):
		fout = open(filename, 'w')
		node_num = len(self.vectors.keys())
		fout.write("{} {}\n".format(node_num, self.size))
		for node, vec in self.vectors.items():
			fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
		fout.close()

    ## 以某个词为开始随机游走
	def randomWalk(self,_g, _corpus_num, _deep_num, _current_word):
		_corpus = []
		for i in range(_corpus_num):
			sentence = [_current_word]
			current_word = _current_word
			count = 0
			while count<_deep_num:
				count+=1
				_node_list = []
				_weight_list = []
				for _nbr, _data in _g[current_word].items():
					_node_list.append(_nbr)
					_weight_list.append(_data['weight']) #初始weight全为1,可以添加
				_ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
				sel_node = self.roulette(_node_list, _ps)
				sentence.append(sel_node)
				current_word = sel_node
			_corpus.append(sentence)
		return _corpus

    #选择下一个结点
	def roulette(self,_datas, _ps):
		return np.random.choice(_datas, p=_ps)

   #随机游走生成句子列表
	def prduce_sentence(self):
		self.sentences = []
		for word in self.g.nodes():
			corpus = self.randomWalk(self.g, self.num, self.deep_num, word)
			self.sentences += corpus

    #word2vec向量表示
	def word2vec(self):
		model = gensim.models.Word2Vec(self.sentences, sg=1, size=self.size, alpha=0.025, window=3, min_count=1,
									   max_vocab_size=None)

		# outfile = './test'
		# #fname = './testmodel-0103'
		# # save
		# #model.save(fname)
		# #model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
		#model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

		self.vectors = {}
		for node in self.g.nodes():
			self.vectors[node]=model.wv[node]
