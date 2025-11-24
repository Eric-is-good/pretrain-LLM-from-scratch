from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import torch
import sys

# sys.path.append("model/") # 你的模型路径

# --- 模型和分词器加载 (保持不变) ---
# 注意：Streamer 需要 tokenizer 来解码
model = AutoModelForCausalLM.from_pretrained("chat/", trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("chat/")
model.eval()

# --- Prompt 定义 (保持不变) ---
chat_prompt = """<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
<think>"""

chat_prompt_nothink = """<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""

# --- 输入选择 (保持不变) ---
# input_text = "请问你是谁？"
input_text = "守护文化遗产的重要性体现在哪些方面？"
# input_text = "请问什么是算法公平性？"
# input_text = "翻译成中文：Cherry blossom viewing, known as 'hanami' in Japanese, is far more than a casual spring outing."
# input_text = "小明去文具店买学习用品，一支钢笔售价 12 元，一本笔记本售价 5 元。他买了 2 支钢笔和 3 本笔记本，付款时店员给他减免了 3 元。请问小明最终需要支付多少钱？"

# 格式化输入
formatted_text = chat_prompt.format(input=input_text)
# formatted_text = chat_prompt_nothink.format(input=input_text)

inputs = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids.to("cuda")

# --- 【修改核心】：引入 Streamer 实现流式输出 ---

# 1. 定义 Streamer
# skip_prompt=True 表示不打印问题，只打印回答；skip_special_tokens=True 自动过滤特殊符号
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 2. 准备生成参数
generate_kwargs = dict(
    input_ids=input_ids,
    max_length=2048,           # 稍微调大一点，防止还没说完就截断
    do_sample=True,
    temperature=1.0,
    top_k=20,
    top_p=0.95,
    num_return_sequences=1,    # 【重要】流式输出必须为 1，否则多个句子混在一起无法阅读
    streamer=streamer
)

# 3. 启动多线程
# model.generate 是阻塞运行的，所以必须放在子线程里，主线程才能去“接”吐出来的字
thread = Thread(target=model.generate, kwargs=generate_kwargs)
thread.start()

print("\n--- 正在生成 (Streaming) ---\n")
if "<think>" in formatted_text:
    print("<think>", end="", flush=True) # 手动补一个开头，因为 prompt 里的可能被 skip 掉

# 4. 主线程循环打印
for new_text in streamer:
    # end="" 防止自动换行，flush=True 强制立即刷新缓冲区，实现“蹦字”效果
    print(new_text, end="", flush=True)

print("\n\n--- 生成结束 ---")
