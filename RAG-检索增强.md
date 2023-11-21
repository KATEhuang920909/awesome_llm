### 简介

检索增强生成（RAG，Retrieval Augmented Generation），用来解决大模型的三类问题而提出的概念，先对现有文档做检索，而非由LLM任意发挥，然后基于检索结果生成答案。目前较为流行的是基于LangChain检索问答框架。

### 大模型的三类问题

```
原始大模型生成适合较为发散的问题和通用知识点问答，比如生成一篇文章、感冒了怎么办等等。但是存在如下三类问题：
```

```
1.幻觉问题：大模型的预训练语料里面肯定不可能覆盖全部知识点，相关领域知识点具有极强的专业性，大模型可能会一本正经的胡说八道。
```

```
2.时效问题：大模型的预训练具有迟滞性，及时信息不能覆盖到，譬如公司上一年财报内容，增长率等信息，大模型通常不会输出正确的结果。
```

```
3.数据安全问题：若预训练模型语料中包含商业数据信息如经营数据、合同文件等，在生成的时候可能会泄露商业数据，存在信息安全风险，解决该问题最好的方法就是把敏感信息放在公司本地，实现大模型的本地化部署。
```

### RAG架构

![图片](https://mmbiz.qpic.cn/mmbiz_png/vDwtLC7WmgTaNqq28jYCO9njn7ibGhJtMZQecrp5p5iaXsOUCjBTDBf9BGoOKa37k57hl4YeaHhSrSJ6ZLcd54Ow/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

流程为：

```
数据提取>>embedding>>创建索引>>检索>>自动排序>>LLM归纳生成。
```

### 技术细节

#### 概览

![图片](https://mmbiz.qpic.cn/mmbiz_png/vDwtLC7WmgTaNqq28jYCO9njn7ibGhJtMegp8YnUOgagpibAxU7b7P5OqGy2w8nvWcRqWpiayZ8R6pybzUe9aOOlA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

#### 1.数据索引

+ 数据提取

  * 数据加载：包括数据加载、提取PDF、EXCEL、及数据库或者API等。
  * 数据清洗：将无关数据剔除、数据格式化等。
  * 元数据提取：提取关键信息，如人名、事件、时间、标题等等。
