# Graph-Embedding


# dataset
* Cora (citation dataset)
cora.cites 
<paper_id1> <paper_id2>
文献之间的引用关系 后一个引用前一个 "paper2->paper1"

cora.centent
<paper_id><出现的词为0，不出现为1><文献类别>
一共七个类别
Case_Based
Genetic_Algorithms
Neural_Networks
Probabilistic_Methods
Reinforcement_Learning
Rule_Learning
Theory

注：向量表示之后可以分个类，看与ground_truth的差异
只要我们的方法比以往好就行

## deepwalk
```
python run_deepwalk.py
随机游走获得词向量方法
```
## TADW
```
python run_tadw.py
加入每个结点文本特征的词向量方法
```

