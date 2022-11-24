
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM 
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

text = """def _get_repo_functions(root, supported_file_extensions, relevant_node_types):
    functions = []
    print('Extracting functions from {}'.format(root))
    for fp in tqdm([root + '/' + f for f in os.popen('git -C {} ls-files'.format(root)).read().split('\n')]):
        if not os.path.isfile(fp):
            continue
        with open(fp, 'r') as f:
            lang = supported_file_extensions.get(fp[fp.rfind('.'):])
            if lang:
                parser = get_parser(lang)
                file_content = f.read()
                tree = parser.parse(bytes(file_content, 'utf8'))
                all_nodes = list(_traverse_tree(tree.root_node))
                functions.extend(_extract_functions(
                    all_nodes, fp, file_content, relevant_node_types))
    return functions
"""


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
        original_embeddings = embeddings(original, tokenizer, model, embeddings_cache)
        originals_embeddings.append(original_embeddings)
        
    for idx in idxs:
        inputs.input_ids[torch.tensor(0), torch.tensor(idx)] = tokenizer.mask_token_id
    encoder_output = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    mask_token_logits = encoder_output.logits[0, mask_token_index, :]

    # this is slow AF????
    prob = nnf.softmax(mask_token_logits, dim=1)
    top_ps, _ = prob.topk(1, dim = 1)
    
    preds = []
    preds_embeddings = []
    tops = torch.topk(mask_token_logits, 1, dim=1)
    for i in range(len(idxs)):
        pred = tokenizer.decode(tops.indices[i].tolist()) 
        emb  = embeddings(pred, tokenizer, model, embeddings_cache)
        preds.append(pred)
        preds_embeddings.append(emb)
        
    out = []
    for i in range(len(idxs)):
        similarity = cosine_similarity(originals_embeddings[i].detach().numpy(), preds_embeddings[i].detach().numpy())
        out.append({
            'idx': idxs[i],
            'original': originals[i],
            'predicted': preds[i],
            'cosine_similarity': similarity,
            'probability': top_ps[i].item(),
        })
    return out

def process_text(text, mask_ratio=0.1):
    model_name = "microsoft/unixcoder-base"
    device = 'cpu'
   
    config = RobertaConfig.from_pretrained(model_name)
    config.is_decoder = False

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name, config=config)

    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.weight = model.base_model.embeddings.word_embeddings.weight
    model.lm_head = lm_head

    embeddings_cache = {}
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    out = []

    ln = len(inputs['input_ids'][0])
    n_masks = int(mask_ratio * ln)
    last = None
    for i in tqdm(range(int(ln / n_masks))):
        idxs = []
        for j in range(n_masks):
            val = (j*  int(ln / n_masks) + i)
            idxs.append(val)
            last = val
        out = out + for_idx(idxs, deepcopy(inputs), model, tokenizer, embeddings_cache)
        
    for i in tqdm(range(last+1, ln, 1)):
        out = out + for_idx([i], deepcopy(inputs), model, tokenizer, embeddings_cache)
        
    return sorted(out, key=lambda x: x['idx'])

def choose_color(token):
    if token['original'].strip() != token['predicted'].strip():
        if token['cosine_similarity'] < 0.9:
            if token['probability'] > 0.5:
                return 'text-red-500'
            else:
                return 'text-stone-300'
    
def prep_for_rendering(processed_text):
    out = processed_text[1:-1] # skip start and end tokens
    return [ {**o, **{'text_color': choose_color(o)}} for o in out]

def run(text):
    processed_text = process_text(text)
        
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("index.html.j2")
    content = template.render(tokens=prep_for_rendering(processed_text), file_name='foo.py')
    with open("index.html", "w") as f:
        f.write(content)
    url = 'file://' + os.getcwd() + '/index.html'
    webbrowser.open(url) 

def main():
    parser = argparse.ArgumentParser(
        prog='sus', description='Detetects suspicious code in a given file')
    parser.add_argument('file', nargs='?', default='foo.py')
    args = parser.parse_args()
    
    with open(args.file, 'r') as f:
        text = f.read()
        run(text)


if __name__ == '__main__':
    main()