from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import logging
import torch.nn.functional as nnf
from transformers import pipeline
import numpy as np

from copy import deepcopy
import torch
import torch.nn as nn

text = """
def main():
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
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

def embeddings(text, tokenizer, model, cache):
    if text in cache:
        return cache[text]
    else:
        inputs = tokenizer(text, return_tensors='pt')
        encoder_out = model(**inputs, output_hidden_states=True)
        embeddings = encoder_out.hidden_states[0][0, 1, :]
        cache[text] = embeddings
        return embeddings
    
def for_idx(idxs, inputs, model, tokenizer, embeddings_cache):
    originals = []
    originals_embeddings = []
    for idx in idxs:
        original = tokenizer.decode(inputs['input_ids'][0][idx])
        originals.append(original)
        original_embeddings = embeddings(
            original, tokenizer, model, embeddings_cache)
        originals_embeddings.append(original_embeddings)

    for idx in idxs:
        inputs.input_ids[torch.tensor(0), torch.tensor(
            idx)] = tokenizer.mask_token_id
    encoder_output = model(**inputs)
    mask_token_index = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id)[1]

    mask_token_logits = encoder_output.logits[0, mask_token_index, :]

    prob = nnf.softmax(mask_token_logits, dim=1)
    top_ps, _ = prob.topk(1, dim=1)

    preds = []
    preds_embeddings = []
    tops = torch.topk(mask_token_logits, 1, dim=1)
    for i in range(len(idxs)):
        pred = tokenizer.decode(tops.indices[i].tolist())
        emb = embeddings(pred, tokenizer, model, embeddings_cache)
        preds.append(pred)
        preds_embeddings.append(emb)

    out = []
    for i in range(len(idxs)):
        similarity = cosine_similarity(
            originals_embeddings[i].detach().numpy(), preds_embeddings[i].detach().numpy())
        out.append({
            'idx': idxs[i],
            'original': originals[i],
            'predicted': preds[i],
            'cosine_similarity': similarity,
            'probability': top_ps[i].item(),
        })
    return out

inputs = tokenizer(text, return_tensors='pt',
                   truncation=True, max_length=1024)
out = []
embeddings_cache = {}

out = out + for_idx([7, 8, 9, 10, 11, 12, 13], deepcopy(inputs), model,
                    tokenizer, embeddings_cache)

for x in out:
    print(x)
