面向智能问答的篇章排序是由搜狗搜索和CCIR 2017联合主办的智能问答评测比赛。比赛任务设定为给定问题下对候选答案篇章进行排序。本评测提供智能问答领域的大规模公开数据集，包含5万条来自互联网真实用户问答需求的问题，和50万条人工标注的候选篇章文本。

http://huodong.sogou.com/sogou_ccir_qa/

NOTINCE: tensorflow 版本务必使用 0.11.0rc2，否则请修改text_nn.py中代码

很遗憾没有进入前十，最终成绩为 NDCG@3:0.6959 NDCG@5:0.7371 ，只差0.17%

![image](https://github.com/colourful-tree/MyTensorflow/blob/master/ccir/image/score.jpeg)

篇章排序任务一般可以用Learning to Rank（简写为LTR）的方法来解决。但是本次比赛中训练集问题答案句对仅被标注为3类，并没有给出指定的排序信息。导致在运用LTR算法进行排序时偏序关系不明显。
另一方面本次比赛评测标准采用nDCG。只要算法可以将候选句对中与问题高相关的找出来并排在前面即可，并不需要关心同样是高相关性的候选句对之间的先后关系。所以将此次篇章排序任务转化为文本分类任务，综合利用attention基础的Bi-RNN和CNN的特点分别对问题与答案进行embedding表示。最后综合考虑两种网络结果给出分类结果。此外，发现将分类结果转化为排序时，加入分类结果的置信度可以提高系统nDCG得分。

网络结构图:


![image](https://github.com/colourful-tree/MyTensorflow/blob/master/ccir/image/struct.png)

准确率:


![image](https://github.com/colourful-tree/MyTensorflow/blob/master/ccir/image/acc.png)

题外话：
前后两个月的评测结束了，结果比较唏嘘，有一些遗憾。
一方面很好奇前十的队伍有没有同样将这个问题当做分类问题来做的，另一方面在工作之外利用业余时间来实现一些interesting的事情也蛮有趣的，从拿到数据，分析数据，清洗数据，做baseline，想办法，分析网络结构，改变网络结构，分析tensorboard，改变参数等等等等，也学到了很多东西，也没有什么遗憾了。
（或许用一些工程方法提高一点点badcase的召回就能获得更好的结果了）。

荞麦记于2017.07.04
