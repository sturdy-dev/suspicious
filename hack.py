
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

def for_idx(idx, inputs, model, tokenizer, embeddings_cache):
    original = tokenizer.decode(inputs['input_ids'][0][idx])
    original_embeddings = embeddings(original, tokenizer, model, embeddings_cache)

    inputs.input_ids[torch.tensor(0), torch.tensor(idx)] = tokenizer.mask_token_id
    encoder_output = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    mask_token_logits = encoder_output.logits[0, mask_token_index, :]
    predicted = tokenizer.decode(torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist())
    predicted_embeddings = embeddings(predicted, tokenizer, model, embeddings_cache)
    
    similarity = cosine_similarity(original_embeddings.detach().numpy(), predicted_embeddings.detach().numpy())
    return {
        'original': original, 
        'predicted': predicted, 
        'cosine_similarity': similarity}

def process_text(text):
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
    inputs = tokenizer(text, return_tensors='pt')
    out = []
    for idx in tqdm(range(len(inputs['input_ids'][0]))):
        out.append(for_idx(idx, deepcopy(inputs), model, tokenizer, embeddings_cache))
    return out

def similarity_to_color(similarity):
    if similarity < 0.5:
        return 'text-red-500'
    elif similarity < 0.9:
        return 'text-red-400'
    
def prep_for_rendering(processed_text):
    out = processed_text[1:-1] # skip start and end tokens
    return [ {**o, **{'text_color': similarity_to_color(o['cosine_similarity'])}} for o in out]

def main():
    processed_text = None
    if os.path.exists('.testout.p'):
        print('Loading processed text from file')
        with open('.testout.p', 'rb') as f:
            processed_text = pickle.load(f)
    else:
        processed_text = process_text(text)
        pickle.dump(processed_text, open('.testout.p', 'wb'))
        
        
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("index.html.j2")
    content = template.render(tokens=prep_for_rendering(processed_text), file_name='foo.py')
    with open("index.html", "w") as f:
        f.write(content)
    url = 'file://' + os.getcwd() + '/index.html'
    webbrowser.open(url) 
    # print(content)


if __name__ == '__main__':
    main()