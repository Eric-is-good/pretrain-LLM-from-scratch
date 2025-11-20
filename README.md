# pretrain-LLM-from-scratch

# 手把手从0训练大语言模型



我们的模型叫 福尔摩斯（Holmes）

训练相关报告（知乎） https://www.zhihu.com/column/c_1834334189455011840

福尔摩斯采用的理念可以用他的一句话概括：“人脑就像一间小阁楼，杂物堆得越多，越难找到有用的东西。”他认为只应保留那些与侦探工作有关的知识，而其他不相关的信息会占用他的“心智空间”。因此，他选择忽略了“地球绕太阳转”这类科学常识。

福尔摩斯为了自己的演绎法推理能力，甚至舍弃了地球绕着太阳转的常识。
我们也希望制作一个专注数学和推理能力的小模型，大胆舍弃其他一切可以舍弃的。

模型特点：我们的模型是原生思维链模型，将思维链内化到模型本身，即在预训练过程中教会模型思维链式思考方式。

<br><br>

## v2 模型 MOE 模型（默认 main 分支）

### 模型特征（[知乎链接](https://zhuanlan.zhihu.com/p/1948409709209031905)）：

- 模型大小 0.6 B，激活0.2b，训练 token 数 100 B
- 复现了 Deepseek 的 [MLA（论文2.1.1节）](https://arxiv.org/pdf/2412.19437) 和 美团的 [MOE（论文2.1节）](https://arxiv.org/abs/2509.01322)，以及 deepseek 中提到的[两个负载均衡方法（论文2.1.2节）](https://arxiv.org/pdf/2412.19437)
- 我们希望通过这次预训练，掌握 MOE 模型的训练技巧


### 模型地址

从 [**这里 huggingface**](https://huggingface.co/ej2/Holmes_moe_history) 下载模型权重，我们记录了一系列的模型权重点和训练日志。

*目前最新的是训练了 70B token pretrain的模型。*

pretrain 阶段暂告一段落，现在开始 SFT



### 数据集

https://loz42cxvnh.feishu.cn/docx/ObsMd55ufomENvxWodKciCaSnCc 

https://huggingface.co/datasets/ej2/Holmes_moe_sft_data



### 完成代码功能

#### tokenizing 和 预训练代码，一行代码启动训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 pretrain.py
```



#### 基于 LLaMA Factory 的微调代码，位于 SFT 子文件夹下

```bash
bash train.sh
```



#### 补全测试代码（针对预训练模型的补全任务）

下载 hf 上面的模型权重文件夹后，运行 demo 即可

```bash
python3 continue_writing_demo.py 
```



#### 对话测试代码（针对微调模型的对话任务，使用 qwen3 混合思考模板）

```bash
python3 chat_demo.py 
```





<br><br><br><br><br><br>



## v1 模型稠密模型（需要切换到 v1 分支）

### 模型特征：

- 模型大小 0.5 B，训练 token 数 0.5 T
- 我们希望模型在架构上，使用改良的 transformer，使得模型在推理时，具有更快的速度
- 我们希望模型具有强大的思维链能力（接近高一量级 ≈ 10B的模型所具有的）


### 数据集
我们的**特色**数据集[增强方案](https://github.com/JustinLiii/Holmes_DataAug)



### 完成代码功能

1. PT 预训练
2. SFT 微调
3. GRPO 强化学习



### 如何运行

从 [**这里 huggingface**](https://huggingface.co/ej2/Holmes_history/tree/main) 下载模型权重，我们记录了一系列的模型权重点。

*目前最新的是训练了 150B token chat的模型。*

将下载的 model.safetensors 放进代码 model 文件夹下，即可运行 chat_demo.py 文件，就可以对话啦！！！！

