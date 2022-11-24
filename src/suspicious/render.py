from jinja2 import Template, Environment, FileSystemLoader
import webbrowser
import os
import pkg_resources


def _choose_color(token):
    if token['original'].strip() != token['predicted'].strip():
        if token['cosine_similarity'] < 0.9:
            if token['probability'] > 0.8:
                return 'text-red-500'
            elif token['probability'] > 0.5:
                return 'text-red-300'
            else:
                return 'text-stone-300'


def _prep_for_rendering(processed_text):
    out = [token for token in processed_text if token['original'].strip(
    ) != '<s>' and token['original'].strip() != '</s>']
    return [{**o, **{'text_color': _choose_color(o)}} for o in out]


def render(tokens, file_name):
    templates = pkg_resources.resource_filename(__name__, "templates")
    environment = Environment(loader=FileSystemLoader(templates))
    template = environment.get_template("index.html.j2")
    content = template.render(
        tokens=_prep_for_rendering(tokens), file_name=file_name)
    with open("index.html", "w") as f:
        f.write(content)
    url = 'file://' + os.getcwd() + '/index.html'
    webbrowser.open(url)
