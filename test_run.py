import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct"

print("Loading model and tokenizer...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model (Keeping the V100 eager attention fix!)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16,   # Standard 16-bit precision for V100
    attn_implementation="eager"  # Bypasses FlashAttention
)

# Set up the chat pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Create a test prompt using the chat format
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain the concept of fine-tuning a language model in two short sentences."}
]

print("\nGenerating response...")
# Generate the output
output = pipe(
    messages,
    max_new_tokens=100,
    return_full_text=False,
    temperature=0.7,
    do_sample=True,
)

print("\n=== PHI-3 OUTPUT ===")
print(output[0]['generated_text'])
print("====================\n")
