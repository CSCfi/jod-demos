import flask
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import (StringField, RadioField,
                     SelectField, SelectMultipleField,
                     IntegerRangeField)
from wtforms.validators import DataRequired, Optional

import numpy as np
import pandas as pd
from scipy.special import softmax

import json
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

cfg = read_config()
debug = cfg['debug']

assert cfg['do_tfidf'] or cfg['do_ft'] or cfg['do_strans'], (
    "At least one algorithm needed")

if cfg['lemmatizer'] == "tnpp":
    from lemmatizer_tnpp import lemmatize, test_lemmatizer
elif cfg['lemmatizer'] == "voikko":
    from lemmatizer_voikko import lemmatize, test_lemmatizer
elif cfg['lemmatizer'] == "snowball":
    from stemmer_snowball import lemmatize, test_lemmatizer
else:
    assert 0, "Unknown lemmatizer: "+cfg['lemmatizer']

if cfg['do_tfidf']:
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
    vectorizer7 = joblib.load('{}/tmt/avo/{}-{}-tfidf.pkl'.format(cfg['datadir'], cfg['dataset7'], cfg['lemmatizer']))
    X_tfidf7 = joblib.load('{}/tmt/avo/{}-{}-tfidf-mat.pkl'.format(cfg['datadir'], cfg['dataset7'], cfg['lemmatizer']))
else:
    print('Skipping TF-IDF')

if cfg['do_ft']:
    print('Loading FastText model:', cfg['ftmodel'])
    model_ft = ft.load_model(cfg['ftmodel'])
    X_ft1 = joblib.load('{}/tmt/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset1'], cfg['lemmatizer']))
    X_ft2 = joblib.load('{}/tmt/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset2'], cfg['lemmatizer']))
    X_ft3 = joblib.load('{}/esco/fi/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset3'], cfg['lemmatizer']))
    X_ft4 = joblib.load('{}/esco/fi/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset4'], cfg['lemmatizer']))
    X_ft5 = joblib.load('{}/konfo/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset5'], cfg['lemmatizer']))
    X_ft6 = joblib.load('{}/konfo/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset6'], cfg['lemmatizer']))
    X_ft7 = joblib.load('{}/tmt/avo/{}-{}-fasttext.pkl'.format(cfg['datadir'], cfg['dataset7'], cfg['lemmatizer']))
else:
    print('Skipping FastText')

if cfg['do_strans']:
    print('Loading Sentence Transformer model:', cfg['stmodel'])
    model_strans = SentenceTransformer(cfg['stmodel'])
    Xemb1 = np.load(cfg['datadir']+'/tmt/'+cfg['embfile1'])
    Xemb2 = np.load(cfg['datadir']+'/tmt/'+cfg['embfile2'])
    Xemb3 = np.load(cfg['datadir']+'/esco/fi/'+cfg['embfile3'])
    Xemb4 = np.load(cfg['datadir']+'/esco/fi/'+cfg['embfile4'])
    Xemb5 = np.load(cfg['datadir']+'/konfo/'+cfg['embfile5'])
    Xemb6 = np.load(cfg['datadir']+'/konfo/'+cfg['embfile6'])
    Xemb7 = np.load(cfg['datadir']+'/tmt/avo/'+cfg['embfile7'])
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
df5 = df5.rename(columns={"Unnamed: 0": "id"})
df6 = pd.read_csv('{}/konfo/{}.csv'.format(cfg['datadir'], cfg['dataset6']))
df6 = df6.set_index('nimi-fi')
df6 = df6.rename(columns={"Unnamed: 0": "id"})
df7 = pd.read_csv('{}/tmt/avo/{}.csv'.format(cfg['datadir'], cfg['dataset7']),
                  converters={'avo-attributes-int': pd.eval,
                              'avo-restrictions-int': pd.eval,
                              'avo-code-int': pd.eval})
df7 = df7.set_index('name')

with open('{}/tmt/avo/avo-fields.json'.format(cfg['datadir']), encoding='utf-8') as fh:
    avo_fields = json.load(fh)
with open('{}/tmt/avo/avo-attribs.json'.format(cfg['datadir']), encoding='utf-8') as fh:
    avo_attribs = json.load(fh)
with open('{}/tmt/avo/avo-restrs.json'.format(cfg['datadir']), encoding='utf-8') as fh:
    avo_restrs = json.load(fh)
avo_riasec = ["R (realistic, käytännöllinen)", "I (investigative, tieteellinen)",
              "A (artistic, taiteellinen)", "S (social, sosiaalinen)",
              "E (enterprising, yrittävä)", "C (conventional, systemaattinen)"]

df7['avo-field-int'] = df7['avo-field-int'].astype(int)
df7_attrib = df7['avo-attributes-int'].apply(pd.Series)
df7_attrib = df7_attrib.fillna(-1).astype(int)
df7_restr = df7['avo-restrictions-int'].apply(pd.Series)
df7_restr = df7_restr.fillna(-1).astype(int)
df7_riasec = df7['avo-code-int'].apply(pd.Series)

print('Testing lemmatizer:', cfg['lemmatizer'])
test_lemmatizer()

print('All done')

# ----------------------------------------------------------------------

def get_edutxt(educ, tamm, tamk, t_yo):
    if debug:
        print(educ, tamm, tamk, t_yo)
    txt = None
    if educ == "amm" and tamm > -1:
        txt = df2[df2['id'] == tamm]['kuvaus-fi-nohtml'].values[0]
    elif educ == "amk" and tamk > -1:
        txt = df5[df5['id'] == tamk]['kuvaus-fi'].values[0]
    elif educ == "yo" and t_yo > -1:
        txt = df6[df6['id'] == t_yo]['kuvaus-fi'].values[0]
    return txt

def _tfidf(txt_int, txt_edu, txt_ski, w, vect, X):
    query_vec_int = vect.transform([txt_int])
    res = cosine_similarity(X, query_vec_int).squeeze()
    weight = (np.exp(w)-1)/(np.exp(5)-1)
    if txt_edu is not None:
        query_vec_edu = vect.transform([txt_edu])
        res += weight*cosine_similarity(X, query_vec_edu).squeeze()
    if txt_ski is not None:
        query_vec_ski = vect.transform([txt_ski])
        res += weight*cosine_similarity(X, query_vec_ski).squeeze()
    return res

def get_tfidf(txt_int, txt_edu=None, txt_ski=None, weighting=5):
    return [_tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer1, X_tfidf1),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer2, X_tfidf2),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer3, X_tfidf3),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer4, X_tfidf4),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer5, X_tfidf5),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer6, X_tfidf6),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer7, X_tfidf7)]

def _fasttext(q_int, q_edu, q_ski, w, X):
    res = cosine_similarity(X, q_int).squeeze()
    weight = (np.exp(w)-1)/(np.exp(5)-1)
    if q_edu is not None:
        res += weight*cosine_similarity(X, q_edu).squeeze()
    if q_ski is not None:
        res += weight*cosine_similarity(X, q_ski).squeeze()
    return res

def get_fasttext(txt_int, txt_edu=None, txt_ski=None, weighting=5):
    query_ft_int = model_ft.get_sentence_vector(txt_int)
    query_ft_int = query_ft_int.reshape(1, -1)
    if txt_edu is not None:
        query_ft_edu = model_ft.get_sentence_vector(txt_edu)
        query_ft_edu = query_ft_edu.reshape(1, -1)
    else:
        query_ft_edu = None
    if txt_ski is not None:
        query_ft_ski = model_ft.get_sentence_vector(txt_ski)
        query_ft_ski = query_ft_ski.reshape(1, -1)
    else:
        query_ft_ski = None

    return [_fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft1),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft2),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft3),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft4),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft5),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft6),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft7)]

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def _strans(q_int, q_edu, q_ski, w, X):
    res = np.zeros(len(X))
    for i in range(len(X)):
        res[i] = cos_sim(q_int, X[i])
    weight = (np.exp(w)-1)/(np.exp(5)-1)
    if q_edu is not None:
        for i in range(len(X)):
            res[i] += weight*cos_sim(q_edu, X[i])
    if q_ski is not None:
        for i in range(len(X)):
            res[i] += weight*cos_sim(q_ski, X[i])
    return res

def get_strans(txt_int, txt_edu=None, txt_ski=None, weighting=5):
    qemb_int = model_strans.encode(txt_int)
    if txt_edu is not None:
        qemb_edu = model_strans.encode(txt_edu)
    else:
        qemb_edu = None
    if txt_ski is not None:
        qemb_ski = model_strans.encode(txt_ski)
    else:
        qemb_ski = None

    return [_strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb1),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb2),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb3),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb4),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb5),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb6),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb7)]

def _get_result(res, educ_level, fields, attributes, restrictions, riasec):
    if res is not None:
        r1 = pd.Series(res[0], index=df1.index).add_suffix(' (ammattitieto)')
        r2 = pd.Series(res[1], index=df2['nimi-fi']).add_suffix(' (eperusteet)')
        r3 = pd.Series(res[2], index=df3['preferredLabel'])
        r4 = pd.Series(res[3], index=df4['preferredLabel']).add_suffix(' (esco)')
        r5 = pd.Series(res[4], index=df5.index).add_suffix(' (konfo amk)')
        r6 = pd.Series(res[5], index=df6.index).add_suffix(' (konfo yo)')
        r7 = pd.Series(res[6], index=df7.index)
        #r_occ = pd.concat([r1, r4])
        r_occ = r7

        if educ_level == "amm":
            r_edu = r5
            r_occ = r_occ*df7['avo-perustaso']
        elif educ_level == "amk":
            r_edu = r6
            r_occ = r_occ*df7['avo-amk-taso']
        elif educ_level == "yo":
            r_edu = r6
            r_occ = r_occ*df7['avo-yliopistotaso']
        else:
            r_edu = pd.concat([r2, r5, r6])

        if '-1' not in fields:
            fields = [int(f) for f in fields]
            df7['field-tmp'] = df7['avo-field-int'].isin(fields)
            if debug:
                print('Ammattialat:', len(df7[df7['field-tmp']]), fields)
                print(" ".join(list(df7[df7['field-tmp']].index)))
            r_occ = r_occ*(1.0 + 1.0*df7['field-tmp'])

        if '-1' not in attributes:
            attributes = [int(a) for a in attributes]
            df7['attrib-tmp'] = df7_attrib.isin(attributes).any(axis=1)
            if debug:
                print('Työn sisältö:', len(df7[df7['attrib-tmp']]), attributes)
                print(" ".join(list(df7[df7['attrib-tmp']].index)))
            r_occ = r_occ*(1.0 + 1.0*df7['attrib-tmp'])

        if '-1' not in restrictions:
            restrictions = [int(r) for r in restrictions]
            df7['restr-tmp'] = df7_restr.isin(restrictions).any(axis=1)
            if debug:
                print('Rajoitukset:', len(df7[df7['restr-tmp']]), restrictions)
                print(" ".join(list(df7[df7['restr-tmp']].index)))
            r_occ = r_occ*(~df7['restr-tmp'])

        if riasec != ('-1', '-1'):
            riasec = tuple(map(int, riasec))
            df7['riasec-tmp'] = df7_riasec.isin(riasec).any(axis=1)
            if debug:
                print('RIASEC:', len(df7[df7['riasec-tmp']]), riasec)
                print(" ".join(list(df7[df7['riasec-tmp']].index)))
            r_occ = r_occ*(1.0 + 1.0*df7['riasec-tmp'])

        r_occ = r_occ.add_suffix(' (avo)')

        return {'education': r_edu.sort_values(ascending=False).index.tolist()[:5],
                'occupations': r_occ.sort_values(ascending=False).index.tolist()[:5]}
    else:
        return {'education': ["-"]*5, 'occupations': ["-"]*5}

def get_results(res_tfidf, res_fasttext, res_strans,
                educ_level="lukio", fields=['-1'], attributes=['-1'],
                restrictions=['-1'], riasec=(-1, -1)):

    tfidfdict = _get_result(res_tfidf, educ_level, fields, attributes, restrictions, riasec)
    fasttextdict = _get_result(res_fasttext, educ_level, fields, attributes, restrictions, riasec)
    stransdict = _get_result(res_strans, educ_level, fields, attributes, restrictions, riasec)

    if all(r is not None for r in [res_tfidf, res_fasttext, res_strans]):
        res_combined = []
        for i in range(7):
            res_combined.append(np.mean(np.array([res_tfidf[i], res_fasttext[i],
                                                  res_strans[i]]), axis=0))
    else:
        res_combined = None

    combineddict = _get_result(res_combined, educ_level, fields, attributes, restrictions, riasec)

    return {'tfidf': tfidfdict, 'fasttext': fasttextdict,
            'strans': stransdict, 'combined': combineddict}

# ----------------------------------------------------------------------

class MyForm(FlaskForm):

    weighting = IntegerRangeField('Kiinnostus vs. osaaminen:', default=5)

    name = StringField('Kiinnostus:', validators=[DataRequired()])

    educ = RadioField('Koulutus:',
                      choices=[('lukio','lukio'),('amm','ammattikoulu'),
                               ('amk','AMK'),('yo','yliopisto')])

    tammlist = [(-1,"-")]
    tammlist.extend([(x['id'], x['nimi-fi']) for _, x in df2.sample(20).iterrows()])
    tamm = SelectField(u'Ammattikoulututkinto:',
                       choices=tammlist, default=-1)

    tamklist = [(-1,"-")]
    tamklist.extend([(x['id'], xi) for xi, x in df5.sample(20).iterrows()])
    tamk = SelectField(u'AMK-tutkinto:',
                       choices=tamklist, default=-1)

    t_yolist = [(-1,"-")]
    t_yolist.extend([(x['id'], xi) for xi, x in df6.sample(20).iterrows()])
    t_yo = SelectField(u'Yliopistotutkinto:',
                       choices=t_yolist, default=-1)

    skills = StringField('Muu osaaminen:', validators=[Optional()])

    afielist = [(-1,"-")]
    afielist.extend([(xi, x) for xi, x in enumerate(avo_fields)])
    afie = SelectMultipleField(u'Ammattialat:',
                               choices=afielist, default=-1)

    aattlist = [(-1,"-")]
    aattlist.extend([(xi, x) for xi, x in enumerate(avo_attribs)])
    aatt = SelectMultipleField(u'Työn sisältö:',
                               choices=aattlist, default=-1)

    areslist = [(-1,"-")]
    areslist.extend([(xi, x) for xi, x in enumerate(avo_restrs)])
    ares = SelectMultipleField(u'Rajoitukset:',
                               choices=areslist, default=-1)

    arialist = [(-1,"-")]
    arialist.extend([(xi, x) for xi, x in enumerate(avo_riasec)])
    aria = SelectField(u'RIASEC:', choices=arialist, default=-1)
    ari2 = SelectField(u'RIASEC 2:', choices=arialist, default=-1)

# ----------------------------------------------------------------------

@app.route("/",methods=["GET"])
def parse_get():
    global p
    txt=flask.request.args.get("text")
    if not txt:
        txt = "pidän lentämisestä ja lentokoneista"
    txt_lem = lemmatize(txt)

    res = get_results(get_tfidf(txt_lem) if cfg['do_tfidf'] else None,
                      get_fasttext(txt_lem) if cfg['do_ft'] else None,
                      get_strans(txt) if cfg['do_strans'] else None)

    form = MyForm(meta={'csrf': False})
    return flask.Response(flask.render_template("index2.html",
                                                lemmatized=txt_lem,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=res['combined'],
                                                form=form, debug=debug),
                          mimetype="text/html; charset=utf-8")

@app.route("/",methods=["POST"])
def parse_post():
    form = MyForm(meta={'csrf': False})
    txt = form.name.data
    if not txt:
        return """Error occurred""", 400
    txt_lem = lemmatize(txt)

    txt_edu = get_edutxt(form.educ.data, int(form.tamm.data),
                         int(form.tamk.data), int(form.t_yo.data))
    txt_edu_lem = lemmatize(txt_edu)

    txt_ski = form.skills.data
    txt_ski_lem = lemmatize(txt_ski)

    res = get_results(get_tfidf(txt_lem, txt_edu_lem, txt_ski_lem, form.weighting.data) if cfg['do_tfidf'] else None,
                      get_fasttext(txt_lem, txt_edu_lem, txt_ski_lem, form.weighting.data) if cfg['do_ft'] else None,
                      get_strans(txt, txt_lem, txt_ski_lem, form.weighting.data) if cfg['do_strans'] else None,
                      form.educ.data, form.afie.data, form.aatt.data,
                      form.ares.data, (form.aria.data, form.ari2.data))

    return flask.Response(flask.render_template("index2.html",
                                                lemmatized=txt_lem,
                                                lemmatized_skills=txt_ski_lem,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=res['combined'],
                                                form=form, debug=debug),
                          mimetype="text/html; charset=utf-8")

# ----------------------------------------------------------------------
