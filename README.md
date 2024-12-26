# pretrain-LLM-from-scratch

# 手把手从0训练大语言模型



我们的模型叫 福尔摩斯（Holmes）

训练相关报告（知乎） https://www.zhihu.com/column/c_1834334189455011840

福尔摩斯采用的理念可以用他的一句话概括：“人脑就像一间小阁楼，杂物堆得越多，越难找到有用的东西。”他认为只应保留那些与侦探工作有关的知识，而其他不相关的信息会占用他的“心智空间”。因此，他选择忽略了“地球绕太阳转”这类科学常识。

福尔摩斯为了自己的演绎法推理能力，甚至舍弃了地球绕着太阳转的常识。
我们也希望制作一个专注数学和推理能力的小模型，大胆舍弃其他一切可以舍弃的。

模型特点：我们的模型是原生思维链模型，将思维链内化到模型本身，即在预训练过程中教会模型思维链式思考方式。

模型特征：

- 模型大小 0.5 B，训练 token 数 0.5 T
- 我们希望模型在架构上，使用改良的 transformer，使得模型在推理时，具有更快的速度
- 我们希望模型具有强大的思维链能力（接近高一量级 ≈ 10B的模型所具有的）



## 如何运行

从 [**这里 huggingface**](https://huggingface.co/ej2/Holmes_history/tree/main) 下载模型权重，我们记录了一系列的模型权重点。

*目前最新的是训练了 150B token 的模型。*

将下载的 model.safetensors 放进代码 model 文件夹下，即可运行 some_test.py 文件，目前只有预训练模型，只能补全句子。
