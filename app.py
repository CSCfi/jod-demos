import flask
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField
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

from lemmatizer import lemmatize, test_lemmatizer

from yamlconfig import read_config

app = flask.Flask(__name__)
Bootstrap(app)

class MyForm(FlaskForm):
    name = StringField('hakulause:', validators=[DataRequired()])

DO_TFIDF, DO_FT, DO_STRANS = True, False, False
assert DO_TFIDF or DO_FT or DO_STRANS, "At least one algorithm needed"

cfg = read_config()

if DO_TFIDF:
    print('Loading TF-IDF models')
    vectorizer1 = joblib.load('{}/tmt/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset1'], cfg['lemmatizer']))
    X_tfidf1 = joblib.load('{}/tmt/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset1'], cfg['lemmatizer']))
    vectorizer2 = joblib.load('{}/tmt/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset2'], cfg['lemmatizer']))
    X_tfidf2 = joblib.load('{}/tmt/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset2'], cfg['lemmatizer']))
    vectorizer3 = joblib.load('{}/esco/fi/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset3'], cfg['lemmatizer']))
    X_tfidf3 = joblib.load('{}/esco/fi/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset3'], cfg['lemmatizer']))
    vectorizer4 = joblib.load('{}/esco/fi/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset4'], cfg['lemmatizer']))
    X_tfidf4 = joblib.load('{}/esco/fi/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset4'], cfg['lemmatizer']))
    vectorizer5 = joblib.load('{}/konfo/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset5'], cfg['lemmatizer']))
    X_tfidf5 = joblib.load('{}/konfo/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset5'], cfg['lemmatizer']))
    vectorizer6 = joblib.load('{}/konfo/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset6'], cfg['lemmatizer']))
    X_tfidf6 = joblib.load('{}/konfo/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset6'], cfg['lemmatizer']))
else:
    print('Skipping TF-IDF')

if DO_FT:
    print('Loading FastText model:', cfg['ftmodel'])
    model_ft = ft.load_model(cfg['ftmodel'])
    X_ft1 = joblib.load('{}/tmt/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset1'], cfg['lemmatizer']))
    X_ft2 = joblib.load('{}/tmt/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset2'], cfg['lemmatizer']))
    X_ft3 = joblib.load('{}/esco/fi/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset3'], cfg['lemmatizer']))
    X_ft4 = joblib.load('{}/esco/fi/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset4'], cfg['lemmatizer']))
    X_ft5 = joblib.load('{}/konfo/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset5'], cfg['lemmatizer']))
    X_ft6 = joblib.load('{}/konfo/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset6'], cfg['lemmatizer']))
else:
    print('Skipping FastText')

if DO_STRANS:
    print('Loading Sentence Transformer model:', cfg['stmodel'])
    model_strans = SentenceTransformer(cfg['stmodel'])
    Xemb1 = np.load(cfg['datadir']+'/tmt/'+cfg['embfile1'])
    Xemb2 = np.load(cfg['datadir']+'/tmt/'+cfg['embfile2'])
    Xemb3 = np.load(cfg['datadir']+'/esco/fi/'+cfg['embfile3'])
    Xemb4 = np.load(cfg['datadir']+'/esco/fi/'+cfg['embfile4'])
    Xemb5 = np.load(cfg['datadir']+'/konfo/'+cfg['embfile5'])
    Xemb6 = np.load(cfg['datadir']+'/konfo/'+cfg['embfile6'])
else:
    print('Skipping Sentence Transformer')

print('Loading datasets')
df1 = pd.read_csv('{}/tmt/{}.csv'.format(cfg['datadir'], cfg['dataset1']))
df1 = df1.set_index('name')
df2 = pd.read_csv('{}/tmt/{}.csv'.format(cfg['datadir'], cfg['dataset2']), index_col=0)
df3 = pd.read_csv('{}/esco/fi/{}'.format(cfg['datadir'], cfg['dataset3']))
df4 = pd.read_csv('{}/esco/fi/{}'.format(cfg['datadir'], cfg['dataset4']))
df5 = pd.read_csv('{}/konfo/{}.csv'.format(cfg['datadir'], cfg['dataset5']))
df5 = df5.set_index('nimi-fi')
df6 = pd.read_csv('{}/konfo/{}.csv'.format(cfg['datadir'], cfg['dataset6']))
df6 = df6.set_index('nimi-fi')

print('Testing lemmatizer:', cfg['lemmatizer'])
test_lemmatizer(cfg['lemmatizer'])

print('All done')

def get_tfidf(txt):
    query_vec1 = vectorizer1.transform([txt])
    res1 = cosine_similarity(X_tfidf1, query_vec1).squeeze()
    query_vec2 = vectorizer2.transform([txt])
    res2 = cosine_similarity(X_tfidf2, query_vec2).squeeze()
    query_vec3 = vectorizer3.transform([txt])
    res3 = cosine_similarity(X_tfidf3, query_vec3).squeeze()
    query_vec4 = vectorizer4.transform([txt])
    res4 = cosine_similarity(X_tfidf4, query_vec4).squeeze()
    query_vec5 = vectorizer5.transform([txt])
    res5 = cosine_similarity(X_tfidf5, query_vec5).squeeze()
    query_vec6 = vectorizer6.transform([txt])
    res6 = cosine_similarity(X_tfidf6, query_vec6).squeeze()
    return [res1, res2, res3, res4, res5, res6]
    
def get_fasttext(txt):
    query_ft = model_ft.get_sentence_vector(txt)
    res1 = cosine_similarity(X_ft1, query_ft.reshape(1, -1)).squeeze()
    res2 = cosine_similarity(X_ft2, query_ft.reshape(1, -1)).squeeze()
    res3 = cosine_similarity(X_ft3, query_ft.reshape(1, -1)).squeeze()
    res4 = cosine_similarity(X_ft4, query_ft.reshape(1, -1)).squeeze()
    res5 = cosine_similarity(X_ft5, query_ft.reshape(1, -1)).squeeze()
    res6 = cosine_similarity(X_ft6, query_ft.reshape(1, -1)).squeeze()
    return [res1, res2, res3, res4, res5, res6]

def cos_sim(a, b): 
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_strans(txt):
    qemb = model_strans.encode(txt)

    res1 = np.zeros(len(Xemb1))
    for i in range(len(Xemb1)):
        res1[i] = cos_sim(qemb, Xemb1[i])
    res2 = np.zeros(len(Xemb2))
    for i in range(len(Xemb2)):
        res2[i] = cos_sim(qemb, Xemb2[i])
    res3 = np.zeros(len(Xemb3))
    for i in range(len(Xemb3)):
        res3[i] = cos_sim(qemb, Xemb3[i])
    res4 = np.zeros(len(Xemb4))
    for i in range(len(Xemb4)):
        res4[i] = cos_sim(qemb, Xemb4[i])
    res5 = np.zeros(len(Xemb5))
    for i in range(len(Xemb5)):
        res5[i] = cos_sim(qemb, Xemb5[i])
    res6 = np.zeros(len(Xemb6))
    for i in range(len(Xemb6)):
        res6[i] = cos_sim(qemb, Xemb6[i])
    return [res1, res2, res3, res4, res5, res6]

def get_results(res_tfidf, res_fasttext, res_strans):
    if res_tfidf is not None:
        df1['res_tfidf'] = res_tfidf[0]
        df2['res_tfidf'] = res_tfidf[1]
        df3['res_tfidf'] = res_tfidf[2]
        df4['res_tfidf'] = res_tfidf[3]
        df5['res_tfidf'] = res_tfidf[4]
        df6['res_tfidf'] = res_tfidf[5]
        tfidfdict = {'ammattitieto': df1.sort_values('res_tfidf', ascending=False).index.tolist()[:5],
                     'eperusteet': df2.sort_values('res_tfidf', ascending=False)['nimi-fi'].tolist()[:5],
                     'esco-taidot': df3.sort_values('res_tfidf', ascending=False)['preferredLabel'].tolist()[:5],
                     'esco-ammatit': df4.sort_values('res_tfidf', ascending=False)['preferredLabel'].tolist()[:5],
                     'konfo-amk': df5.sort_values('res_tfidf', ascending=False).index.tolist()[:5],
                     'konfo-yo': df6.sort_values('res_tfidf', ascending=False).index.tolist()[:5]}
    else:
        tfidfdict = {'ammattitieto': ["-"]*5, 'eperusteet': ["-"]*5,
                     'esco-taidot': ["-"]*5, 'esco-ammatit': ["-"]*5,
                     'konfo-amk': ["-"]*5, 'konfo-yo': ["-"]*5}

        
    if res_fasttext is not None:
        df1['res_fasttext'] = res_fasttext[0]
        df2['res_fasttext'] = res_fasttext[1]
        df3['res_fasttext'] = res_fasttext[2]
        df4['res_fasttext'] = res_fasttext[3]
        df5['res_fasttext'] = res_fasttext[4]
        df6['res_fasttext'] = res_fasttext[5]
        fasttextdict = {'ammattitieto': df1.sort_values('res_fasttext', ascending=False).index.tolist()[:5],
                        'eperusteet': df2.sort_values('res_fasttext', ascending=False)['nimi-fi'].tolist()[:5],
                        'esco-taidot': df3.sort_values('res_fasttext', ascending=False)['preferredLabel'].tolist()[:5],
                        'esco-ammatit': df4.sort_values('res_fasttext', ascending=False)['preferredLabel'].tolist()[:5],
                        'konfo-amk': df5.sort_values('res_fasttext', ascending=False).index.tolist()[:5],
                        'konfo-yo': df6.sort_values('res_fasttext', ascending=False).index.tolist()[:5]}
    else:
        fasttextdict =  {'ammattitieto': ["-"]*5, 'eperusteet': ["-"]*5,
                         'esco-taidot': ["-"]*5, 'esco-ammatit': ["-"]*5,
                         'konfo-amk': ["-"]*5, 'konfo-yo': ["-"]*5}

    if res_strans is not None:
        df1['res_strans'] = res_strans[0]
        df2['res_strans'] = res_strans[1]
        df3['res_strans'] = res_strans[2]
        df4['res_strans'] = res_strans[3]
        df5['res_strans'] = res_strans[4]
        df6['res_strans'] = res_strans[5]
        stransdict = {'ammattitieto': df1.sort_values('res_strans', ascending=False).index.tolist()[:5],
                      'eperusteet': df2.sort_values('res_strans', ascending=False)['nimi-fi'].tolist()[:5],
                      'esco-taidot': df3.sort_values('res_strans', ascending=False)['preferredLabel'].tolist()[:5],
                      'esco-ammatit': df4.sort_values('res_strans', ascending=False)['preferredLabel'].tolist()[:5],
                      'konfo-amk': df5.sort_values('res_strans', ascending=False).index.tolist()[:5],
                      'konfo-yo': df6.sort_values('res_strans', ascending=False).index.tolist()[:5]}
    else:
        stransdict =  {'ammattitieto': ["-"]*5, 'eperusteet': ["-"]*5,
                       'esco-taidot': ["-"]*5, 'esco-ammatit': ["-"]*5,
                       'konfo-amk': ["-"]*5, 'konfo-yo': ["-"]*5}

    if all(r is not None for r in [res_tfidf, res_fasttext, res_strans]):
        df1['res_combined'] = np.mean(np.array([res_tfidf[0], res_fasttext[0], res_strans[0]]), axis=0)
        df2['res_combined'] = np.mean(np.array([res_tfidf[1], res_fasttext[1], res_strans[1]]), axis=0)
        df3['res_combined'] = np.mean(np.array([res_tfidf[2], res_fasttext[2], res_strans[2]]), axis=0)
        df4['res_combined'] = np.mean(np.array([res_tfidf[3], res_fasttext[3], res_strans[3]]), axis=0)
        df5['res_combined'] = np.mean(np.array([res_tfidf[4], res_fasttext[4], res_strans[4]]), axis=0)
        df6['res_combined'] = np.mean(np.array([res_tfidf[5], res_fasttext[5], res_strans[5]]), axis=0)
        combineddict = {'ammattitieto': df1.sort_values('res_combined', ascending=False).index.tolist()[:5],
                        'eperusteet': df2.sort_values('res_combined', ascending=False)['nimi-fi'].tolist()[:5],
                        'esco-taidot': df3.sort_values('res_combined', ascending=False)['preferredLabel'].tolist()[:5],
                        'esco-ammatit': df4.sort_values('res_combined', ascending=False)['preferredLabel'].tolist()[:5],
                        'konfo-amk': df5.sort_values('res_combined', ascending=False).index.tolist()[:5],
                        'konfo-yo': df6.sort_values('res_combined', ascending=False).index.tolist()[:5]}
    else:
        combineddict =  {'ammattitieto': ["-"]*5, 'eperusteet': ["-"]*5,
                         'esco-taidot': ["-"]*5, 'esco-ammatit': ["-"]*5,
                         'konfo-amk': ["-"]*5, 'konfo-yo': ["-"]*5}

    return {'tfidf': tfidfdict, 'fasttext': fasttextdict,
            'strans': stransdict, 'combined': combineddict}

@app.route("/",methods=["GET"])
def parse_get():
    global p
    txt=flask.request.args.get("text")
    if not txt:
        txt = "pidän lentämisestä ja lentokoneista"
    txt_lem = lemmatize(txt, cfg['lemmatizer'])

    res = get_results(get_tfidf(txt_lem) if DO_TFIDF else None,
                      get_fasttext(txt_lem) if DO_FT else None,
                      get_strans(txt) if DO_STRANS else None)

    form = MyForm(meta={'csrf': False})
    return flask.Response(flask.render_template("index-bs.html",
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
    txt_lem = lemmatize(txt, cfg['lemmatizer'])
    res = get_results(get_tfidf(txt_lem) if DO_TFIDF else None,
                      get_fasttext(txt_lem) if DO_FT else None,
                      get_strans(txt) if DO_STRANS else None)
    return flask.Response(flask.render_template("index-bs.html",
                                                lemmatized=txt_lem,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=res['combined'],
                                                form=form),
                          mimetype="text/html; charset=utf-8")
