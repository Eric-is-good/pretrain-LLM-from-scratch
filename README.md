# pretrain-LLM-from-scratch

# 手把手从0训练大语言模型



名字暂且叫 ***Think Twice***（TT），也可以叫 Rethink Reasoning （RR）

模型特点：我们的模型是原生思维链模型，将思维链内化到模型本身，即在预训练过程中教会模型思维链式思考方式。

模型特征：

- 模型大小 0.5 B，训练 token 数 0.5 T
- 我们希望模型在架构上，使用改良的 transformer，使得模型在推理时，具有更快的速度
- 我们希望模型具有强大的思维链能力（接近高一量级 ≈ 10B的模型所具有的）

