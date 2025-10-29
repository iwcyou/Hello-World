from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型名称
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载模型（自动选择CPU/GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 如果没有GPU可以改为 float32
    device_map="auto"
)

# 3. 输入提示词
prompt = "Explain the difference between supervised and unsupervised learning in one paragraph."

# 4. Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 5. 推理生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9
    )

# 6. 解码输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
