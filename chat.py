import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Zigeng/DMax-Math-16B", trust_remote_code=True, device_map="cuda:0"
)
model = model.to(torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Zigeng/DMax-Math-16B", trust_remote_code=True)

prompt = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?" + "\nLet's think step by step\n"

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)

nfe, generated_tokens = model.generate_spd(
    inputs=input_ids,
    gen_length=2048,
    block_length=32,
    threshold=0.0,
)

generated_answer = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
)

print(generated_answer)
print("nfe:",nfe,"token length",len(generated_tokens[0]))