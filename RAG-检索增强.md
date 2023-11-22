### 简介

检索增强生成（RAG，Retrieval Augmented Generation），用来解决大模型的三类问题而提出的概念，先对现有文档做检索，而非由LLM任意发挥，然后基于检索结果生成答案。目前较为流行的是基于LangChain检索问答框架。

### 大模型的三类问题

​		原始大模型生成适合较为发散的问题和通用知识点问答，比如生成一篇文章、感冒了怎么办等等。但是存在如下三类问题：

​		1.幻觉问题：大模型的预训练语料里面肯定不可能覆盖全部知识点，相关领域知识点具有极强的专业性，大模型可能会一本正经的胡说八道。

​		2.时效问题：大模型的预训练具有迟滞性，及时信息不能覆盖到，譬如公司上一年财报内容，增长率等信息，大模型通常不会输出正确的结果。

​		3.数据安全问题：若预训练模型语料中包含商业数据信息如经营数据、合同文件等，在生成的时候可能会泄露商业数据，存在信息安全风险，解决该问题最好的方法就是把敏感信息放在公司本地，实现大模型的本地化部署。

### RAG架构

![图片](https://mmbiz.qpic.cn/mmbiz_png/vDwtLC7WmgTaNqq28jYCO9njn7ibGhJtMZQecrp5p5iaXsOUCjBTDBf9BGoOKa37k57hl4YeaHhSrSJ6ZLcd54Ow/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

流程为：

```tex
数据提取>>embedding>>创建索引>>检索>>自动排序>>LLM归纳生成。
```

### 技术细节

#### 概览（参考）

![图片](https://mmbiz.qpic.cn/mmbiz_png/vDwtLC7WmgTaNqq28jYCO9njn7ibGhJtMegp8YnUOgagpibAxU7b7P5OqGy2w8nvWcRqWpiayZ8R6pybzUe9aOOlA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

#### 1.数据索引

+ 数据提取

  * 数据加载：包括数据加载、提取PDF、EXCEL、及数据库或者API等。
  * 数据清洗：将无关数据剔除、数据格式化等。
  * 元数据提取：提取关键信息，如人名、事件、时间、标题、关键短语等等。

+ 分块（Chunking）

  + 硬分块：固定token大小为512/256进行分块
  + 软分块：硬分块可能会将最后一句切割开来，导致句子不完整，因此将512缩减为480，留下“缓冲地带”加载完整句子。
  + 语义分块：具体的文档中可能有换行符，或者明显的切分符号，一般在切分符内语义相同，考虑实际情况可以用语义分块。
  + 其他分块：适用于特殊场景，如表格内容的分块、小标题分块等等，需要分析数据，总结分块规律。

+ 向量化

  向量化是将文本、图像、语音等转化成向量矩阵的过程，embedding的好坏能直接影响检索的质量，特别是相关度。目前有如下几种方法实现数据的向量化

  * 开源模型：如[腾讯词向量](https://ai.tencent.com/ailab/nlp/en/download.html)、[text2vec](https://huggingface.co/shibing624/text2vec-base-chinese)
  * 自己训练模型：目前成熟的方法有Sentence-BERT；SimCSE、ConSERT等对比学习等

#### 2.检索环节

​		检索环节技术含量很高，也可以做的非常细，涉及到query/document的前处理，检索技术和策略如es检索、向量检索、多路检索，重排序，查询轮换等。

* query转换

  query中可能含有大量和检索信息无关的内容，如“鲁迅是中国现代伟大的作家，社会活动家，新文化运动的发起人之一，写了《狂人日记》、《孔乙己》等小说散文，请问鲁迅的出生日期和去世日期分别是哪一年？”  。这里若将长问句检索可能会出现无关信息，因此，我们需要将上述query转化成“鲁迅的出生日期和去世日期”，套用模板句法分析或者大模型sft将query重写，即可以实现query转换。

* 检索技术和策略

  * 词义检索：可以基于关键词匹配检索、结合bm25方法检索，可以用elastic-search做检索
  * 语义向量检索：构建document向量化索引，常见的索引结构有HNSW、LSH、PQ、树索引等，结合余弦向量进行检索，常用的框架有Faiss、Milvus。
  * 多路检索：多路检索可以多字段检索、结合词义和语义检索。
  * 检索策略：将document分类，然后再构建多个索引。query前处理的时候确定是哪个类型然后在对应的索引下检索，可以提升检索效率和准确度。

* 重排序

  ​		结合query与检索结果的匹配度进行重排序，然后取topk个，最终送入LLM处理，重排序可以复用检索匹配值，也可以训练一个排序模型，排序模型有point-wise和pair-wise两种方法。

* 查询轮换：这是查询检索的一种方式，一般会有几种方式：

* - 子查询：可以在不同的场景中使用各种查询策略，比如可以使用LlamaIndex等框架提供的查询器，采用树查询（从叶子结点，一步步查询，合并），采用向量查询，或者最原始的顺序查询chunks等；
  - HyDE：这是一种抄作业的方式，生成相似的或者更标准的 prompt 模板**。**

#### 3.生成

​		将前述的检索内容作为query的上下文补充信息，从而生成最终结果。实际上也是prompt工程。

​		这里仍然可以结合指令微调的方式微调大模型。通过这种方式，使 LLM 更好地利用相关背景知识，并训练 LLM 即使在检索错误块的情况下也能产生准确的预测，从而使模型能够依赖自己的知识。



### 现有工具

langchain  :https://python.langchain.com/docs/use_cases/question_answering/

LlamaIndex:https://gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html



参考资料：[大模型主流应用RAG的介绍——从架构到技术细节](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/FqyaTK2Mb4VJolK81P5_1g)
