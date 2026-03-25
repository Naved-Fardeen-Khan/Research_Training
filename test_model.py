import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. SETTINGS ---
base_model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "./phi-3-mini-lora/final_adapter"

# VARIABLES 
age_in_months = 6
topic = "birds"
num_generations = 10

print(f"Loading base model and adapter for a {age_in_months}-month-old...")

# --- 2. LOAD MODEL & ADAPTER ---
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)

# Load LoRA adapter on top of the base model
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# --- 3. GENERATION LOOP ---
# We use the same formatting as the training data
messages = [
    {"role": "user", "content": f"Generate a sentence of child-directed speech for a {age_in_months}-month-old infant about {topic}."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
prompt_length = inputs.input_ids.shape[1]
print(f"\n--- Generating {num_generations} speeches for a {age_in_months}-month-old ---")
print(f"\nPrompt: {messages[0]['content']}")
# Output parameters:
temperature = 0.8 # Higher temperature means more randomness in the output
top_p = 0.9       # Top-p sampling means the model will only consider the smallest set of tokens whose cumulative probability exceeds top_p.

print(f"\nUsing parameters: temperature={temperature}, top_p={top_p}\n")

for i in range(num_generations):
    with torch.no_grad():
        # Output settings:
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=temperature,
            do_sample=True,    # Must be True to get different results each time
            top_p=top_p,        
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    new_tokens = outputs[0][prompt_length:]
    speech = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"{i+1}. {speech}")