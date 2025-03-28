## *大模型微调实验报告**

### 实验目标

梳理大模型微调方法，评估各种基座和微调方法的实验效果。

### 基础模型

\1.Llama

\2.Qwen

\3.Chatglm4

\4.

### 微调策略

#### LoRA系列

低秩适配（LoRA）的核心思想是冻结原始参数，通过低秩分解引入可训练参数。

LoRA假设预训练模型的参数矩阵的更新可以表示为一个低秩矩阵。具体来说，对于一个预训练好的权重矩阵W(d\*e)，LoRA引入两个低秩矩阵A(d\*r)和B(r*e)，其中r是秩，远小于d和e向量为度。在微调过程中，只需优化A和B，而W保持不变。更新后的权重矩阵为：

W0=W+BA

![img](https://picx.zhimg.com/v2-ddc35d4db9800b82351bd0a38ec22889_1440w.jpg)

伪代码示例：

```python
importtorch
importtorch.nnasnn

classLoRALayer(nn.Module):
def__init__(self,in_dim,out_dim,rank):
super().__init__()
self.A=nn.Parameter(torch.randn(in_dim,rank))
self.B=nn.Parameter(torch.zeros(rank,out_dim))
self.rank=rank**#秩的大小**
defforward(self,x):
returnx@(self.A@self.B)**#低秩矩阵乘积
```

关键技术演进

![img](https://mmbiz.qpic.cn/mmbiz_jpg/iceGibVicRfib5kEOC7QWhtb8RJ1RnMpNpiaLXFiaN2sb8nNkjibiauNnbFL5q5uicP6iafkicsTInOURaK1bqJeaVg6uBmYg/640?wx_fmt=jpeg&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

**AdaLoRA**：

- **核心思想**：AdaLoRA旨在**动态调整各层的秩分配**。
- **实现方式**：AdaLoRA通过一些指标（例如，梯度范数）来评估不同层的重要性，并根据重要性动态地调整LoRA的秩。更重要的层分配更高的秩，从而获得更好的性能。

**QLoRA**：

- **核心思想**：QLoRA将**4-bit量化**与LoRA相结合，以进一步降低显存占用。
- **实现方式**：QLoRA首先将预训练模型的权重**量化**为4-bit精度，然后在此基础上应用LoRA。由于4-bit量化可以显著降低显存占用，因此QLoRA可以在有限的GPU资源上微调更大的模型。
- **显存节省**：QLoRA可以节省高达70%的显存。

**Delta-LoRA**：

- **核心思想**：Delta-LoRA引入**参数更新量的动量机制**。
- **实现方式**：Delta-LoRA在更新LoRA参数时，考虑之前的更新方向和幅度，从而更稳定地进行微调。

#### 提示微调技术

**提示微调（Prompt-Tuning）**是一种通过**设计合适的提示（Prompt）**来引导预训练模型完成下游任务的技术。与全参数微调和LoRA不同，提示微调通常**不直接修改预训练模型的参数**（注意不是完全不修改参数），而是通过优化提示相关的向量来调整模型的行为。

**核心思想**：

- **人工设计到可学习**：提示工程（Prompt-Engineering）经历了从人工设计提示到可学习提示的演进过程。
- **利用预训练知识**：通过优化提示，引导模型利用预训练知识，从而减少对标注数据的依赖。

**位置选择策略**：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/iceGibVicRfib5kEOC7QWhtb8RJ1RnMpNpiaLKO6MVDAiaGD4y10bdYT4cwFIiaCRtTwzIiapd8gTOSeG9DuWTEbr1N5Zw/640?wx_fmt=jpeg&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

**Prefix-Tuning**：

![img](https://pica.zhimg.com/v2-7881125bcfb6748b6601d3c1f989f274_1440w.jpg)

- **核心思想**：只在**每层**的**开头**插入可训练的提示向量。在Prefix层加了MLP结构，然后切分为key和value矩阵，然后和每一层的key-value矩阵拼接。训练完成后，只保留Prefix的参数。

- **数学形式**：
  $$
  h^l = Transformer([Prefix^l;x^l])
  $$
  

- **参数量**：每层新增参数d*r（d为维度，r为前缀长度）

- **优点**：Prefix-Tuning可以有效地影响模型的每一层，从而更好地调整模型的行为。

- **缺点**：Prefix-Tuning需要插入大量的提示向量，可能会增加计算成本。

**P-Tuning v2**：

- **核心思想**：**分层插入**位置**可学习**。
- **实现方式**：P-Tuning v2首先将提示向量插入到不同的层中，然后通过训练来确定每个提示向量的最佳位置。
- **数学形式**：
- **优点**：P-Tuningv2可以更灵活地调整提示向量的位置，从而更好地适应不同的任务。

**Prompt-Tuning**（蓝色）：

- **核心思想**：仅在**输入层**添加可训练提示词。
- **参数量**：仅需（为提示词数量）
- **数学形式**：
- **优点**：Prompt-Tuning的实现简单，计算成本低。
- **缺点**：Prompt-Tuning的效果可能不如Prefix-Tuning和P-Tuningv2。



### **实验设计**

**数据集**

数据来源与规模（如领域专用语料、人工标注数据）

来自于CHIP2023-PromptCBLUE-参数高效微调赛道数据集，原始数据集有15个任务，包括分类、ner、总结、生成等基本任务，本实验选取其中三类任务，描述如下：

|       |     任务类型      |   数据集    | 数量 |
| :---: | :---------------: | :---------: | :--: |
| train |        cls        | IMCS-V2-DAC | 1536 |
| train |        ner        |  CMeEE-V2   | 1492 |
| train | report_generation | IMCS-V2-MRG | 872  |
| valid |        cls        | IMCS-V2-DAC | 547  |
| valid |        ner        |  CMeEE-V2   | 496  |
| valid | report_generation | IMCS-V2-MRG | 269  |

数据预处理（清洗、格式转换）

格式转换为json，字段包括instruction、input、output、history，样例如下：

```json
{"instruction": "根据对话内容，判断最后一句话的意图类别是：\n问诊对话历史：\n患者：就是早晨起床干咳的那种，咳一阵\n医生：化验过肺炎支原体感染吗\n患者：不是支原体感染，化验过\n医生：肺炎治疗后，复查过吗\n患者：挂完水医生建议不用继续挂水，吃药，吃了一个星期的金振口服液，匹多莫得，盐酸西替利滴剂还有睡前咀嚼的\n患者：一个星期吃这么多，后来又去医院复查，医生说就吃匹多莫得和睡前咀嚼的就可以了，\n患者：但是现在白天也咳嗽了\n医生：目前看医生给用了抗气道过敏药物\n医生：这个对于改善目前气道高反应状态，还是很对症。\n医生：目前考虑按疗程治疗后，基本的炎症，都已经消下去了\n医生：目前咳嗽，还主要是一个气道高反应，过敏状态\n患者：那怎么办呢\n患者：过敏症状吃什么呢？\n患者：滴剂要不要继续吃\n医生：可以做做雾化\n选择：关于病因的回答，关于注意事项的提问，关于用药建议的提问，关于用药建议的解答，关于症状的回答，关于个人基本信息的回答，关于已有检查和治疗的回答，关于病因的询问，关于就医建议的解答，关于注意事项的解答，给出诊断，关于已有检查和治疗的提问，关于症状的询问，关于就医建议的提问，关于个人基本信息的询问\\","input":"", "target": "关于用药建议的解答", "history": ""}

```

**环境配置**

1.硬件配置（GPU型号、显存占用）

1. linux ubuntu
2. nvidia v100 32G



2.软件配置、环境依赖

 	1. peft==0.14.0
 	2. transformers==4.49.0
	3. trl==0.15.2
	4. torch==2.4.0
	5. bitsandbytes==0.45.3

***\*基本参数\****

优化器与超参数（学习率、batch size、epoch数）

| 超参数                      | 描述                                    | 备注          |
| --------------------------- | --------------------------------------- | ------------- |
| num_train_epochs            | 10                                      |               |
| lora_target                 | 'q_proj,k_proj,v_proj,out_proj,fc1,fc2' | lora系列      |
| learning_rate               | 1e-3                                    |               |
| use_ntk                     | linear                                  |               |
| padding_side                | left                                    |               |
| torch_dtype                 | float16                                 |               |
| quantization                | bnb                                     |               |
| max_input_token             | 2048                                    |               |
| use_firefly_loss            | True                                    |               |
| gradient_accumulation_steps | 4                                       |               |
| optim                       | adamw_torch                             |               |
| lr_scheduler_type           | cosine                                  |               |
| num_virtual_tokens          | 20                                      | prompt_tuning |
| prompt_encoder_hidden_size  | 128                                     | prompt_tuning |



### **实验结果与分析**

**性能对比**



| 基座 | 微调方法 | rouge-1    | rouge-2    | rouge-l    |
| ---- | -------- | ---------- | ---------- | ---------- |
| Qwen | qlora    | 71.758     | 59.341     | 11.761     |
| Qwen | lora     | **73.526** | **60.519** | 11.897     |
| Qwen | prefix   | 71.328     | 58.594     | **17.857** |
| Qwen | prompt   | 59.118     | 43.109     | 10.048     |
| Qwen | ptuning  | 71.122     | 57.709     | 11.529     |





Qwen上lora最优