from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import pipeline
import os
import torch

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

GENERATION_TITLE= "<p style='font-size: 16px; color: white;'>Generated code:</p>"

tokenizer_fim = AutoTokenizer.from_pretrained("bigcode/santacoder",  padding_side="left")

tokenizer_fim.add_special_tokens({
  "additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
  "pad_token": EOD,
})

device="cpu"

tokenizer = AutoTokenizer.from_pretrained("bigcode/christmas-models")
model = AutoModelForCausalLM.from_pretrained("bigcode/christmas-models", trust_remote_code=True).to(device)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def extract_fim_part(s: str):
    # Find the index of 
    start = s.find(FIM_MIDDLE) + len(FIM_MIDDLE)
    stop = s.find(EOD, start) or len(s)
    return s[start:stop]

def infill(prefix_suffix_tuples, max_new_tokens, temperature):
    if type(prefix_suffix_tuples) == tuple:
        prefix_suffix_tuples = [prefix_suffix_tuples]
        
    prompts = [f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}" for prefix, suffix in prefix_suffix_tuples]
    # `return_token_type_ids=False` is essential, or we get nonsense output.
    inputs = tokenizer_fim(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    # WARNING: cannot use skip_special_tokens, because it blows away the FIM special tokens.
    return [        
        extract_fim_part(tokenizer_fim.decode(tensor, skip_special_tokens=False)) for tensor in outputs
    ]
    
def post_processing_fim(prefix, middle, suffix):
    return f"{prefix}{middle}{suffix}"

def fim_generation(prompt, max_new_tokens, temperature):
    prefix = prompt.split("<FILL-HERE>")[0]
    suffix = prompt.split("<FILL-HERE>")[1]
    [middle] = infill((prefix, suffix), max_new_tokens, temperature)
    return post_processing_fim(prefix, middle, suffix)

text = """
def return_hello_world():
<FILL-HERE>
    return message
"""

gen = fim_generation(text, 48, 0.2)

print(gen)
