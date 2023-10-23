from flask import Flask, jsonify, request

import numpy as np
import fasttext as ft
from sentence_transformers import SentenceTransformer

from yamlconfig import read_config

# ----------------------------------------------------------------------

app = Flask(__name__)

cfg = read_config()
debug = True # cfg['debug']

if cfg['lemmatizer'] == "tnpp":
    from lemmatizer_tnpp import lemmatize, test_lemmatizer
elif cfg['lemmatizer'] == "voikko":
    from lemmatizer_voikko import lemmatize, test_lemmatizer
elif cfg['lemmatizer'] == "snowball":
    from stemmer_snowball import lemmatize, test_lemmatizer
else:
    assert 0, "Unknown lemmatizer: "+cfg['lemmatizer']

assert debug or cfg['do_ft'] or cfg['do_strans'], (
    "At least one algorithm needed")

if cfg['do_ft']:
    print('Loading FastText model:', cfg['ftmodel'])
    model_ft = ft.load_model(cfg['ftmodel'])
else:
    print('Skipping FastText')

if cfg['do_strans']:
    print('Loading Sentence Transformer model:', cfg['stmodel'])
    model_strans = SentenceTransformer(cfg['stmodel'])
else:
    print('Skipping Sentence Transformer')

print('All done')

# ----------------------------------------------------------------------

@app.route('/', methods=['GET'])
def get_encodings():
    txt = request.args.get("text")
    print('txt =', txt)
    encodings = {'text': txt}

    txt_lem = lemmatize(txt)
    print('txt_lem =', txt_lem)
    encodings['lemmatized'] = txt_lem

    if debug:
        encodings['random'] = np.random.rand(10).tolist()

    if cfg['do_ft']:
        query_ft = model_ft.get_sentence_vector(txt_lem)
        print('query_ft =', query_ft)
        encodings['query_ft'] = query_ft.tolist()

    if cfg['do_strans']:
        query_strans = model_strans.encode(txt)
        print('query_strans =', query_strans)
        encodings['query_strans'] = query_strans.tolist()

    res = jsonify({'encodings': encodings})
    print(res)
    return res

# ----------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=12123)

# ----------------------------------------------------------------------

