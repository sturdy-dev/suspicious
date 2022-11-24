
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import logging
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pickle
import os
from jinja2 import Template, Environment, FileSystemLoader
import webbrowser
import argparse
import torch.nn.functional as nnf
from math import ceil
import textwrap


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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


def process_batch(inputs, mask_ratio, model, tokenizer, embeddings_cache):
    out = []

    ln = len(inputs['input_ids'][0])
    n_masks = int(mask_ratio * ln)
    for i in tqdm(range(int((ln / n_masks)))):
        idxs = []
        for j in range(n_masks + 1):
            val = (j * int(ln / n_masks) + i)
            if val < ln:
                idxs.append(val)
        out = out + for_idx(idxs, deepcopy(inputs), model,
                            tokenizer, embeddings_cache)

    return out


def process_text(text, mask_ratio=0.1):
    logging.set_verbosity_error()
    model_name = "microsoft/unixcoder-base"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = RobertaConfig.from_pretrained(model_name)
    config.is_decoder = False

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name, config=config)

    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.weight = model.base_model.embeddings.word_embeddings.weight
    model.lm_head = lm_head

    embeddings_cache = {}
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, max_length=1024)
    out = []
    n_batches = int(ceil(len(text)/2500))
    if n_batches > 1:
        print("your file's so big it had to be split in {} batches...".format(n_batches))
    for i in range(0, len(text), 2500):
        text_batch = text[i:i + 2500]
        inputs = tokenizer(text_batch, return_tensors='pt',
                           truncation=True, max_length=1024)
        result = sorted(process_batch(inputs, mask_ratio, model,
                        tokenizer, embeddings_cache), key=lambda x: x['idx'])
        out = out + result

    return out


def choose_color(token):
    if token['original'].strip() != token['predicted'].strip():
        if token['cosine_similarity'] < 0.9:
            if token['probability'] > 0.8:
                return 'text-red-500'
            elif token['probability'] > 0.5:
                return 'text-red-300'
            else:
                return 'text-stone-300'


def prep_for_rendering(processed_text):
    out = [token for token in processed_text if token['original'].strip(
    ) != '<s>' and token['original'].strip() != '</s>']
    return [{**o, **{'text_color': choose_color(o)}} for o in out]


def render(tokens, file_name):
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("index.html.j2")
    content = template.render(
        tokens=prep_for_rendering(tokens), file_name=file_name)
    with open("index.html", "w") as f:
        f.write(content)
    url = 'file://' + os.getcwd() + '/index.html'
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(
        prog='sus', description='Detetects suspicious code in a given file')
    parser.add_argument('file', nargs='?')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        text = f.read()
        tokens = process_text(text)
        render(tokens, args.file)


if __name__ == '__main__':
    main()
