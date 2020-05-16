# HMM 中文词性标注

github地址： https://github.com/eecshope/HMM_POS_Tagging.git

这是一个使用HMM和拉普拉斯平滑进行中文词性标注的简单项目。要运行它，首先请保证正确安装了python 3.7 和hmmlearn

要训练模型，请使用train.py入口
```bash
python train.py
```
要测试模型，请使用test.py入口
```bash
python test.py
```
下面汇报结果：在从训练集抽取出的词表上进行训练和测试，得出结果如下：

|数据集 |准确率 |
| --- | --- |
| 训练集 | 91.20% |
| 测试集 | 83.80% |
本次实验为应对数据稀疏性所采用的平滑方法为拉普拉斯方法。原因在于
-   数据集规模较小，低频词的频数分布式稀疏的，进行插值会造成较大的噪音引入；
-   词表和词性表不算太大；

对于一阶马尔科夫假设，我们记要探测概率的bigram为$(w_i, w_{i+1})$，全词表的大小为$|V|$，那么用$count(x)$代表元素x在训练集中的计数，有

$$
P(w_{i+1}|w_i) = \frac{count(w_i, w_{i+1})+1}{\sum_j^{|V|}count(w_i, x_j)+|V|}
$$
其中$x_j\in V$

我们对转移矩阵和发射概率的每一行都做拉普拉斯平滑，只不过，对于转移矩阵，词表用词性表；而对于发射概率，词表用词汇表。

项目文件夹介绍

- data: python包，内置data_loader模块，用于读取数据并处理成平行的sentence, pos_tags格式
- hmm-dataset: 训练、测试数据集
- model: python包，内置hmm_model模块，主模型
- parameters: 装载训练好的序列化的模型
- train.py: 训练入口
- test.py: 测试入口