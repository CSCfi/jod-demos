import flask
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import (StringField, RadioField,
                     SelectField, SelectMultipleField,
                     IntegerRangeField, BooleanField,
                     FormField, HiddenField, SubmitField)
from wtforms.validators import DataRequired, Optional

import numpy as np
import pandas as pd
from scipy.special import softmax

import os
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
app.config['SECRET_KEY'] = os.urandom(32)
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
df2 = df2.set_index(df2['id'].astype(str) + '-eperusteet')
df3 = pd.read_csv('{}/esco/fi/{}'.format(cfg['datadir'], cfg['dataset3']))
df4 = pd.read_csv('{}/esco/fi/{}'.format(cfg['datadir'], cfg['dataset4']))
df5 = pd.read_csv('{}/konfo/{}.csv'.format(cfg['datadir'], cfg['dataset5']))
df5 = df5.rename(columns={"Unnamed: 0": "id"})
df5 = df5.set_index(df5['id'].astype(str) + '-konfo-amk')
df6 = pd.read_csv('{}/konfo/{}.csv'.format(cfg['datadir'], cfg['dataset6']))
df6 = df6.rename(columns={"Unnamed: 0": "id"})
df6 = df6.set_index(df6['id'].astype(str) + '-konfo-yo')
df7 = pd.read_csv('{}/tmt/avo/{}.csv'.format(cfg['datadir'], cfg['dataset7']),
                  converters={'avo-attributes-int': pd.eval,
                              'avo-restrictions-int': pd.eval,
                              'avo-code-int': pd.eval})
df7 = df7.set_index('uniqueId')

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

df2['row_number'] = range(0, len(df2))
df5['row_number'] = range(0, len(df5))
df6['row_number'] = range(0, len(df6))
df7['row_number'] = range(0, len(df7))

print('Testing lemmatizer:', cfg['lemmatizer'])
test_lemmatizer()

if cfg['do_tfidf'] and cfg['do_ft'] and cfg['do_strans']:
    final_algorithm = 'combined'
elif cfg['do_strans']:
    final_algorithm = 'strans'
elif cfg['do_ft']:
    final_algorithm = 'fasttext'
else:
    final_algorithm = 'tfidf'

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

def _tfidf_fb(fb_pos, fb_neg, suffix, df, X):
    res_fb = np.zeros(len(df))
    for fo in fb_pos:
        if suffix is None or suffix in fo:
            res_fb += cosine_similarity(X, X[df.loc[fo]['row_number']]).squeeze()
    for fo in fb_neg:
        if suffix is None or suffix in fo:
            res_fb -= cosine_similarity(X, X[df.loc[fo]['row_number']]).squeeze()
    return res_fb

def get_tfidf(txt_int, txt_edu=None, txt_ski=None, weighting=5,
              fb_edu_pos=[], fb_edu_neg=[], fb_occ_pos=[], fb_occ_neg=[]):

    res_fb_df2 = _tfidf_fb(fb_edu_pos, fb_edu_neg, '-eperusteet', df2, X_tfidf2)
    res_fb_df5 = _tfidf_fb(fb_edu_pos, fb_edu_neg, '-konfo-amk', df5, X_tfidf5)
    res_fb_df6 = _tfidf_fb(fb_edu_pos, fb_edu_neg, '-konfo-yo', df6, X_tfidf6)
    res_fb_df7 = _tfidf_fb(fb_occ_pos, fb_occ_neg, None, df7, X_tfidf7)

    return [_tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer1, X_tfidf1),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer2, X_tfidf2) + res_fb_df2,
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer3, X_tfidf3),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer4, X_tfidf4),
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer5, X_tfidf5) + res_fb_df5,
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer6, X_tfidf6) + res_fb_df6,
            _tfidf(txt_int, txt_edu, txt_ski, weighting, vectorizer7, X_tfidf7) + res_fb_df7]

def _fasttext(q_int, q_edu, q_ski, w, X):
    res = cosine_similarity(X, q_int).squeeze()
    weight = (np.exp(w)-1)/(np.exp(5)-1)
    if q_edu is not None:
        res += weight*cosine_similarity(X, q_edu).squeeze()
    if q_ski is not None:
        res += weight*cosine_similarity(X, q_ski).squeeze()
    return res

def _fasttext_fb(fb_pos, fb_neg, suffix, df, X):
    res_fb = np.zeros(len(df))
    for fo in fb_pos:
        if suffix is None or suffix in fo:
            res_fb += cosine_similarity(X, X[df.loc[fo]['row_number']].reshape(1, -1)).squeeze()
    for fo in fb_neg:
        if suffix is None or suffix in fo:
            res_fb -= cosine_similarity(X, X[df.loc[fo]['row_number']].reshape(1, -1)).squeeze()
    return res_fb

def get_fasttext(txt_int, txt_edu=None, txt_ski=None, weighting=5,
              fb_edu_pos=[], fb_edu_neg=[], fb_occ_pos=[], fb_occ_neg=[]):

    res_fb_df2 = _fasttext_fb(fb_edu_pos, fb_edu_neg, '-eperusteet', df2, X_ft2)
    res_fb_df5 = _fasttext_fb(fb_edu_pos, fb_edu_neg, '-konfo-amk', df5, X_ft5)
    res_fb_df6 = _fasttext_fb(fb_edu_pos, fb_edu_neg, '-konfo-yo', df6, X_ft6)
    res_fb_df7 = _fasttext_fb(fb_occ_pos, fb_occ_neg, None, df7, X_ft7)

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
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft2) + res_fb_df2,
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft3),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft4),
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft5) + res_fb_df5,
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft6) + res_fb_df6,
            _fasttext(query_ft_int, query_ft_edu, query_ft_ski, weighting, X_ft7) + res_fb_df7]

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

def _strans_fb(fb_pos, fb_neg, suffix, df, X):
    res_fb = np.zeros(len(df))
    for fo in fb_pos:
        if suffix is None or suffix in fo:
            for i in range(len(X)):
                res_fb[i] += cos_sim(X[df.loc[fo]['row_number']], X[i])
    for fo in fb_neg:
        if suffix is None or suffix in fo:
            for i in range(len(X)):
                res_fb[i] -= cos_sim(X[df.loc[fo]['row_number']], X[i])
    return res_fb

def get_strans(txt_int, txt_edu=None, txt_ski=None, weighting=5,
               fb_edu_pos=[], fb_edu_neg=[], fb_occ_pos=[], fb_occ_neg=[]):

    res_fb_df2 = _strans_fb(fb_edu_pos, fb_edu_neg, '-eperusteet', df2, Xemb2)
    res_fb_df5 = _strans_fb(fb_edu_pos, fb_edu_neg, '-konfo-amk', df5, Xemb5)
    res_fb_df6 = _strans_fb(fb_edu_pos, fb_edu_neg, '-konfo-yo', df6, Xemb6)
    res_fb_df7 = _strans_fb(fb_occ_pos, fb_occ_neg, None, df7, Xemb7)

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
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb2) + res_fb_df2,
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb3),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb4),
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb5) + res_fb_df5,
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb6) + res_fb_df6,
            _strans(qemb_int, qemb_edu, qemb_ski, weighting, Xemb7) + res_fb_df7]

def _get_result(res, educ_level, fields, attributes, restrictions, riasec):
    if res is not None:
        r1 = pd.Series(res[0], index=df1.index).add_suffix(' (ammattitieto)')
        r2 = pd.Series(res[1], index=df2.index)
        r3 = pd.Series(res[2], index=df3['preferredLabel'])
        r4 = pd.Series(res[3], index=df4['preferredLabel']).add_suffix(' (esco)')
        r5 = pd.Series(res[4], index=df5.index)
        r6 = pd.Series(res[5], index=df6.index)
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

        #r_occ = r_occ.add_suffix(' (avo)')

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

def id2name(results):
    edunames = []
    for x in results['education']:
        if '-eperusteet' in x:
            name, dataset = df2.loc[x]['nimi-fi'], "eperusteet"
        elif '-konfo-amk' in x:
            name, dataset = df5.loc[x]['nimi-fi'], "konfo-amk"
        elif '-konfo-yo' in x:
            name, dataset = df6.loc[x]['nimi-fi'], "konfo-yo"
        else:
            name, dataset = "???", "???"
        edunames.append("{} ({})".format(name, dataset))
    results.update(education=edunames)

    occnames = ["{} (avo)".format(df7.loc[x]['name']) for x in results['occupations']]
    results.update(occupations=occnames)
    return results

# ----------------------------------------------------------------------

class ItemList(FlaskForm):
    it_0 = HiddenField()
    it_1 = HiddenField()
    it_2 = HiddenField()
    it_3 = HiddenField()
    it_4 = HiddenField()

class ThumbList(FlaskForm):
    th_0 = BooleanField("")
    th_1 = BooleanField("")
    th_2 = BooleanField("")
    th_3 = BooleanField("")
    th_4 = BooleanField("")

class Thumbs(FlaskForm):
    up = FormField(ThumbList)
    down = FormField(ThumbList)

class ResultType(FlaskForm):
    itemlist = FormField(ItemList)
    pos = HiddenField()
    neg = HiddenField()
    thumbs = FormField(Thumbs)

class MyForm(FlaskForm):

    weighting = IntegerRangeField('Kiinnostus vs. osaaminen:', default=5)

    name = StringField('Kiinnostuksen kohteet:',
                       render_kw={'placeholder':
                                  'Kirjoita tähän mitkä alat tai aihepiirit sinua kiinnostavat'},
                       validators=[DataRequired()])

    goal = StringField('Tavoitteet:',
                       render_kw={'placeholder':
                                  'Tähän voit kirjoittaa tavoitteistasi'},
                       validators=[Optional()])

    educ = RadioField('Pohjakoulutus:',
                      choices=[('lukio','lukio'),('amm','ammattikoulu'),
                               ('amk','AMK'),('yo','yliopisto')], default='lukio')

    tammlist = [(-1,"-")]
    tammlist.extend([(x['id'], x['nimi-fi']) for _, x in df2.sample(20).iterrows()])
    tamm = SelectField(u'ammattikoulututkinto:',
                       choices=tammlist, default=-1)

    tamklist = [(-1,"-")]
    tamklist.extend([(x['id'], x['nimi-fi']) for _, x in df5.sample(20).iterrows()])
    tamk = SelectField(u'AMK-tutkinto:',
                       choices=tamklist, default=-1)

    t_yolist = [(-1,"-")]
    t_yolist.extend([(x['id'], x['nimi-fi']) for _, x in df6.sample(20).iterrows()])
    t_yo = SelectField(u'yliopistotutkinto:',
                       choices=t_yolist, default=-1)

    skills = StringField('Muu osaaminen:',
                         render_kw={'placeholder':
                                    'Tähän voit syöttää tietoja muusta osaamisestasi ylläolevan koulutuksen lisäksi'},
                         validators=[Optional()])

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

    edu = FormField(ResultType)
    occ = FormField(ResultType)

    suggest_button = SubmitField(label="Ehdota koulutuksia ja ammatteja")
    restart_button = SubmitField(label="Tyhjennä palautteet")

# ----------------------------------------------------------------------

def items2list(items):
    return [items.it_0, items.it_1, items.it_2, items.it_3, items.it_4]

def thumbs2list(thumbs):
    return [thumbs.th_0, thumbs.th_1, thumbs.th_2, thumbs.th_3, thumbs.th_4]

def update_feedback(thumbs, items, feedback):
        for th, it in zip(thumbs2list(thumbs), items2list(items)):
            if it.data is not None:
                if th.data:
                    feedback.add(it.data)
                elif it.data in feedback:
                    feedback.discard(it.data)

def process_feedback(form):
    if form.suggest_button.data:
        if form.edu.pos.data is not None and len(form.edu.pos.data)>0:
            feedback_edu_pos = set(form.edu.pos.data.split("%"))
        else:
            feedback_edu_pos = set()
        update_feedback(form.edu.thumbs.up, form.edu.itemlist, feedback_edu_pos)

        if form.edu.neg.data is not None and len(form.edu.neg.data)>0:
            feedback_edu_neg = set(form.edu.neg.data.split("%"))
        else:
            feedback_edu_neg = set()
        update_feedback(form.edu.thumbs.down, form.edu.itemlist, feedback_edu_neg)

        if form.occ.pos.data is not None and len(form.occ.pos.data)>0:
            feedback_occ_pos = set(form.occ.pos.data.split("%"))
        else:
            feedback_occ_pos = set()
        update_feedback(form.occ.thumbs.up, form.occ.itemlist, feedback_occ_pos)

        if form.occ.neg.data is not None and len(form.occ.neg.data)>0:
            feedback_occ_neg = set(form.occ.neg.data.split("%"))
        else:
            feedback_occ_neg = set()
        update_feedback(form.occ.thumbs.down, form.occ.itemlist, feedback_occ_neg)

    else:
        feedback_edu_pos = set()
        feedback_edu_neg = set()
        feedback_occ_pos = set()
        feedback_occ_neg = set()

    return (feedback_edu_pos, feedback_edu_neg, feedback_occ_pos, feedback_occ_neg,
            {'edu_pos': len(feedback_edu_pos), 'edu_neg': len(feedback_edu_neg),
             'occ_pos': len(feedback_occ_pos), 'occ_neg': len(feedback_occ_neg)})

# ----------------------------------------------------------------------

def finalize_form(restype, itemlist, fb_pos, fb_neg, continue_fb):

    for i, j in zip(items2list(restype.itemlist), itemlist):
        i.data = j

    restype.pos.data = "%".join(fb_pos)
    restype.neg.data = "%".join(fb_neg)

    for i, j in zip(thumbs2list(restype.thumbs.up), itemlist):
        i.data = True if continue_fb and j in fb_pos else False
    for i, j in zip(thumbs2list(restype.thumbs.down), itemlist):
        i.data = True if continue_fb and j in fb_neg else False

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
    nfeedback = {'edu_pos': 0, 'edu_neg': 0, 'occ_pos': 0, 'occ_neg': 0}
    return flask.Response(flask.render_template(cfg['app3_html_template'],
                                                lemmatized=txt_lem,
                                                nfeedback=nfeedback,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=res[final_algorithm],
                                                form=form, debug=debug),
                          mimetype="text/html; charset=utf-8")

@app.route("/",methods=["POST"])
def parse_post():
    form = MyForm(meta={'csrf': False})
    txt = form.name.data
    if form.goal.data is not None and len(form.goal.data)>3:
        txt = txt + " " + form.goal.data
    if not txt:
        return """Error occurred""", 400
    txt_lem = lemmatize(txt)

    txt_edu = get_edutxt(form.educ.data, int(form.tamm.data),
                         int(form.tamk.data), int(form.t_yo.data))
    txt_edu_lem = lemmatize(txt_edu)

    txt_ski = form.skills.data if form.skills.data is not None and len(form.skills.data)>3 else None
    txt_ski_lem = lemmatize(txt_ski)

    if debug:
        print('form.suggest_button.data:', form.suggest_button.data, type(form.suggest_button.data))
        print('form.restart_button.data:', form.restart_button.data, type(form.restart_button.data))
        print('form.edu.thumbs.up.th_0.data:', form.edu.thumbs.up.th_0.data, type(form.edu.thumbs.up.th_0.data))
        print('form.edu.thumbs.down.th_0.data:', form.edu.thumbs.down.th_0.data, type(form.edu.thumbs.down.th_0.data))
        print('form.edu.itemlist.it_0.data:', form.edu.itemlist.it_0.data, type(form.edu.itemlist.it_0.data))
        print('form.edu.pos.data:', form.edu.pos.data, type(form.edu.pos.data))
        print('form.edu.neg.data:', form.edu.neg.data, type(form.edu.neg.data))
        print('form.occ.thumbs.up.th_0.data:', form.occ.thumbs.up.th_0.data, type(form.occ.thumbs.up.th_0.data))
        print('form.occ.thumbs.down.th_0.data:', form.occ.thumbs.down.th_0.data, type(form.occ.thumbs.down.th_0.data))
        print('form.occ.itemlist.it_0.data:', form.occ.itemlist.it_0.data, type(form.occ.itemlist.it_0.data))
        print('form.occ.pos.data:', form.occ.pos.data, type(form.occ.pos.data))
        print('form.occ.neg.data:', form.occ.neg.data, type(form.occ.neg.data))

    (feedback_edu_pos, feedback_edu_neg,
     feedback_occ_pos, feedback_occ_neg, nfeedback) = process_feedback(form)

    if debug:
        print('feedback_edu_pos', feedback_edu_pos)
        print('feedback_edu_neg', feedback_edu_neg)
        print('feedback_occ_pos', feedback_occ_pos)
        print('feedback_occ_neg', feedback_occ_neg)

    res = get_results(get_tfidf(txt_lem, txt_edu_lem, txt_ski_lem, form.weighting.data,
                                feedback_edu_pos, feedback_edu_neg,
                                feedback_occ_pos, feedback_occ_neg) if cfg['do_tfidf'] else None,
                      get_fasttext(txt_lem, txt_edu_lem, txt_ski_lem, form.weighting.data,
                                feedback_edu_pos, feedback_edu_neg,
                                feedback_occ_pos, feedback_occ_neg) if cfg['do_ft'] else None,
                      get_strans(txt, txt_edu, txt_ski, form.weighting.data,
                                 feedback_edu_pos, feedback_edu_neg,
                                 feedback_occ_pos, feedback_occ_neg) if cfg['do_strans'] else None,
                      form.educ.data, form.afie.data, form.aatt.data,
                      form.ares.data, (form.aria.data, form.ari2.data))

    finalize_form(form.edu, res[final_algorithm]['education'],
                  feedback_edu_pos, feedback_edu_neg, form.suggest_button.data)
    finalize_form(form.occ, res[final_algorithm]['occupations'],
                  feedback_occ_pos, feedback_occ_neg, form.suggest_button.data)

    return flask.Response(flask.render_template(cfg['app3_html_template'],
                                                lemmatized=txt_lem,
                                                lemmatized_skills=txt_ski_lem,
                                                nfeedback=nfeedback,
                                                results1=res['tfidf'],
                                                results2=res['fasttext'],
                                                results3=res['strans'],
                                                results4=id2name(res[final_algorithm]),
                                                form=form, debug=debug),
                          mimetype="text/html; charset=utf-8")

# ----------------------------------------------------------------------
