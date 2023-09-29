import flask
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerRangeField, SubmitField
from wtforms.validators import DataRequired

import numpy as np
import pandas as pd
from scipy.special import softmax

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import fasttext as ft

from sentence_transformers import SentenceTransformer

from yamlconfig import read_config

# ----------------------------------------------------------------------

app = flask.Flask(__name__)
Bootstrap(app)

class MyForm(FlaskForm):
    name = StringField('hakulause:', validators=[DataRequired()])
    weighting1 = IntegerRangeField('kokonaisansiot:', default=0)
    weighting2 = IntegerRangeField('vakanssiaste:', default=0)
    weighting3 = IntegerRangeField('työvoimapula:', default=0)
    weighting4 =  IntegerRangeField('kohtaanto-ongelma:', default=0)
    suggest_button = SubmitField(label="Ehdota ammatteja")

DO_TFIDF, DO_FT, DO_STRANS = True, True, True
assert DO_TFIDF or DO_FT or DO_STRANS, "At least one algorithm needed"

cfg = read_config()
if cfg['lemmatizer'] == "tnpp":
    from lemmatizer_tnpp import lemmatize, test_lemmatizer
elif cfg['lemmatizer'] == "voikko":
    from lemmatizer_voikko import lemmatize, test_lemmatizer
elif cfg['lemmatizer'] == "snowball":
    from stemmer_snowball import lemmatize, test_lemmatizer
else:
    assert 0, "Unknown lemmatizer: "+cfg['lemmatizer']

if DO_TFIDF:
    print('Loading TF-IDF models')
    vectorizer = joblib.load('{}/esco/fi/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset8'], cfg['lemmatizer']))
    X_tfidf = joblib.load('{}/esco/fi/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset8'], cfg['lemmatizer']))
else:
    print('Skipping TF-IDF')

if DO_FT:
    print('Loading FastText model:', cfg['ftmodel'])
    model_ft = ft.load_model(cfg['ftmodel'])
    X_ft = joblib.load('{}/esco/fi/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset8'], cfg['lemmatizer']))
else:
    print('Skipping FastText')

if DO_STRANS:
    print('Loading Sentence Transformer model:', cfg['stmodel'])
    model_strans = SentenceTransformer(cfg['stmodel'])
    Xemb = np.load(cfg['datadir']+'/esco/fi/'+cfg['embfile8'])
else:
    print('Skipping Sentence Transformer')

print('Loading datasets')
df = pd.read_csv('{}/esco/fi/{}'.format(cfg['datadir'], cfg['dataset8']))

print('Processing weightings')
def normalize(ser):
    smax, smin = ser.max(), ser.min()
    return (ser-smin)/(smax-smin)
col1, colnorm1 = 'Kokonaisansion mediaani, e/tunti', 'Kokonaisansion mediaani, normalisoitu'
df[col1] = df[col1].replace(".", np.nan).astype(float)
df[col1] = df[col1].fillna(df[col1].mean())
df[colnorm1] = normalize(df[col1])

col2, colnorm2 = 'Vakanssiaste', 'Vakanssiaste, normalisoitu'
df[col2] = df[col2].clip(upper=0.1)
df[colnorm2] = normalize(df[col2])

col3, colnorm3 = 'Työvoimapula (%)', 'Työvoimapula, normalisoitu'
df[col3] = df[col3].clip(upper=0.05)
df[colnorm3] = normalize(df[col3])

col4, colnorm4 = 'Kohtaanto-ongelma (%)', 'Kohtaanto-ongelma, normalisoitu'
df[col4] = df[col4].clip(upper=0.05)
df[colnorm4] = normalize(df[col4])

print('Testing lemmatizer:', cfg['lemmatizer'])
test_lemmatizer()

print('All done')

# ----------------------------------------------------------------------

def get_weightings(weights):
    return (weights[0]*df[colnorm1] + weights[1]*df[colnorm2] +
            weights[2]*df[colnorm3] + weights[3]*df[colnorm4])

def get_tfidf(txt, w=(0,0,0,0)):
    weights = np.sign(w)*(np.exp(1*np.abs(w))-1)/(np.exp(1*3)-1)
    #print('w:', w, weights)
    query_vec = vectorizer.transform([txt])
    res = (cosine_similarity(X_tfidf, query_vec).squeeze() +
           get_weightings(weights))
    return res

def get_fasttext(txt, w=(0,0,0,0)):
    weights = np.sign(w)*(np.exp(1*np.abs(w))-1)/(np.exp(1*3)-1)
    query_ft = model_ft.get_sentence_vector(txt)
    res = (cosine_similarity(X_ft, query_ft.reshape(1, -1)).squeeze() +
           get_weightings(weights))
    return res

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_strans(txt, w=(0,0,0,0)):
    weights = np.sign(w)*(np.exp(1*np.abs(w))-1)/(np.exp(1*3)-1)
    weightings = get_weightings(weights)
    qemb = model_strans.encode(txt)

    res = np.zeros(len(Xemb))
    for i in range(len(Xemb)):
        res[i] = cos_sim(qemb, Xemb[i]) + weightings[i]
    return res

def get_results(res_tfidf, res_fasttext, res_strans):
    if res_tfidf is not None:
        df['res_tfidf'] = res_tfidf
        dftmp = df.sort_values('res_tfidf', ascending=False).head(10)
        tfidfdict = {'esco-ammatit': dftmp['preferredLabel'].tolist(),
                     'esco-ammatit-extra': ['ka: {:.2f}, va: {:.2f}, tvp: {:.2f}, ko: {:.2f}'
                                            .format(row[colnorm1], row[colnorm2], row[colnorm3], row[colnorm4])
                                            for _, row in dftmp.iterrows()]}
    else:
        tfidfdict = {'esco-ammatit': ["-"]*10, 'esco-ammatit-extra': ["-"]*10}

    if res_fasttext is not None:
        df['res_fasttext'] = res_fasttext
        dftmp = df.sort_values('res_fasttext', ascending=False).head(10)
        fasttextdict = {'esco-ammatit': dftmp['preferredLabel'].tolist(),
                        'esco-ammatit-extra': ['ka: {:.2f}, va: {:.2f}, tvp: {:.2f}, ko: {:.2f}'
                                               .format(row[colnorm1], row[colnorm2], row[colnorm3], row[colnorm4])
                                               for _, row in dftmp.iterrows()]}
    else:
        fasttextdict =  {'esco-ammatit': ["-"]*10, 'esco-ammatit-extra': ["-"]*10}

    if res_strans is not None:
        df['res_strans'] = res_strans
        dftmp = df.sort_values('res_strans', ascending=False).head(10)
        stransdict = {'esco-ammatit': dftmp['preferredLabel'].tolist(),
                      'esco-ammatit-extra': ['ka: {:.2f}, va: {:.2f}, tvp: {:.2f}, ko: {:.2f}'
                                             .format(row[colnorm1], row[colnorm2], row[colnorm3], row[colnorm4])
                                             for _, row in dftmp.iterrows()]}
    else:
        stransdict =  {'esco-ammatit': ["-"]*10, 'esco-ammatit-extra': ["-"]*10}

    if all(r is not None for r in [res_tfidf, res_fasttext, res_strans]):
        df['res_combined'] = np.mean(np.array([res_tfidf, res_fasttext, res_strans]), axis=0)
        dftmp = df.sort_values('res_combined', ascending=False).head(10)
        combineddict = {'esco-ammatit': dftmp['preferredLabel'].tolist(),
                        'esco-ammatit-extra': ['ka: {:.2f}, va: {:.2f}, tvp: {:.2f}, ko: {:.2f}'
                                               .format(row[colnorm1], row[colnorm2], row[colnorm3], row[colnorm4])
                                               for _, row in dftmp.iterrows()]}
    else:
        combineddict =  {'esco-ammatit': ["-"]*10, 'esco-ammatit-extra': ["-"]*10}

    return {'tfidf': tfidfdict, 'fasttext': fasttextdict,
            'strans': stransdict, 'combined': combineddict}

# ----------------------------------------------------------------------

@app.route("/",methods=["GET"])
def parse_get():
    global p
    txt=flask.request.args.get("text")
    if not txt:
        txt = "pidän lentämisestä ja lentokoneista"
    txt_lem = lemmatize(txt)

    res = get_results(get_tfidf(txt_lem) if DO_TFIDF else None,
                      get_fasttext(txt_lem) if DO_FT else None,
                      get_strans(txt) if DO_STRANS else None)

    form = MyForm(meta={'csrf': False})
    return flask.Response(flask.render_template("index5.html",
                                                lemmatized=txt_lem,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=res['combined'],
                                                form=form),
                          mimetype="text/html; charset=utf-8")

@app.route("/",methods=["POST"])
def parse_post():
    form = MyForm(meta={'csrf': False})
    txt = form.name.data
    if not txt:
        return """Error occurred""", 400
    txt_lem = lemmatize(txt)
    weights = (form.weighting1.data, form.weighting2.data,
               form.weighting3.data, form.weighting4.data)
    res = get_results(get_tfidf(txt_lem, weights) if DO_TFIDF else None,
                      get_fasttext(txt_lem, weights) if DO_FT else None,
                      get_strans(txt, weights) if DO_STRANS else None)
    return flask.Response(flask.render_template("index5.html",
                                                lemmatized=txt_lem,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=res['combined'],
                                                form=form),
                          mimetype="text/html; charset=utf-8")
