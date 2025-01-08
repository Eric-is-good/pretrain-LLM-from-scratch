from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "model/"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def chat():
    # 初始化上下文对话内容
    context = []
    print("开始对话 (输入 'exit' 退出):")

    while True:
        # 用户输入
        user_input = input("用户: ")
        if user_input.lower() == 'exit':
            print("对话结束.")
            break

        # 将用户输入加入对话上下文
        context.append({"role": "user", "content": user_input})

        # 生成聊天模板
        text = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        model_inputs = tokenizer(text, return_tensors="pt")
        input_ids = model_inputs.input_ids.to(model.device)

        # 模型生成
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

        # 解码输出
        generated_ids = output[0][len(input_ids[0]):]  # 提取新增部分
        response = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # 打印模型回复
        print("模型:", response.strip())

        # 将模型的回复加入上下文
        context.append({"role": "assistant", "content": \
            response.strip().replace("<|im_end|>","")})

# 启动聊天机器人
if __name__ == "__main__":
    chat()
