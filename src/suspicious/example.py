from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import logging
import torch
import torch.nn as nn

text = """
def <mask>():
    parser = argparse.ArgumentParser(
        prog='sus', description='Detects possibly suspicious stuff in your source files')
    parser.add_argument('file', nargs='?', help='The file to analyze')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        text = f.read()
        tokens = process_text(text)
        render(tokens, args.file)

if __name__ == '__main__':
    main()
"""

def for_idx(idx, inputs, model, tokenizer, embeddings_cache):
    original = tokenizer.decode(inputs['input_ids'][0][idx])
    original_embeddings = embeddings(original, tokenizer, model, embeddings_cache)

    inputs.input_ids[torch.tensor(0), torch.tensor(idx)] = tokenizer.mask_token_id
    predicted = tokenizer.decode(torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist())
    predicted_embeddings = embeddings(predicted, tokenizer, model, embeddings_cache)

    similarity = cosine_similarity(original_embeddings.detach().numpy(), predicted_embeddings.detach().numpy())
    return {
        'original': original, 
        'predicted': predicted, 
        'cosine_similarity': similarity}

logging.set_verbosity_error()
model_name = "microsoft/unixcoder-base-nine"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = RobertaConfig.from_pretrained(model_name)
config.is_decoder = False

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name, config=config)

lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
lm_head.weight = model.base_model.embeddings.word_embeddings.weight
model.lm_head = lm_head

inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, max_length=1024)
    
# idx = 3 # the index of the token to be masked
# original = tokenizer.decode(inputs['input_ids'][0][idx])
# print(original) # main (actual text of token)

# inputs.input_ids[torch.tensor(0), torch.tensor(idx)] = tokenizer.mask_token_id # replacing the token at index idx with the special mask token

encoder_output = model(**inputs)
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

mask_token_logits = encoder_output.logits[0, mask_token_index, :]

predicted = tokenizer.decode(torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist())

print(predicted) # main (priedicted text of token)
