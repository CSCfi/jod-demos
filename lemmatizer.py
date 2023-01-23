# tnpp:
#import requests

# voikko:
from nltk.tokenize import word_tokenize
import libvoikko
v = libvoikko.Voikko(u"fi")

# snowball:
#from nltk.tokenize import word_tokenize
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("finnish")

def lemmatize_using_tnpp(inputstr):
    if inputstr is None:
        return None

    r = requests.post('http://localhost:7689',
                      headers={'Content-Type': 'text/plain', 'charset': 'utf-8'},
                      data=inputstr.encode())
    assert r.status_code == requests.codes.ok

    lemmas = []
    for line in r.text.splitlines():
        if line.startswith('#') or len(line)==0:
            continue
        parts = line.split()
        assert len(parts)>3
        if parts[3] == 'PUNCT':
            continue
        lemma = parts[2].lower().replace("#", "")
        lemmas.append(lemma)
    return " ".join(lemmas)

def lemmatize_using_voikko(inputstr):
    if inputstr is None:
        return None

    tokens = word_tokenize(inputstr)
    lemmas = []
    for t in tokens:
        if len(t)>1:
            a_t = v.analyze(t)
            lemma = a_t[0]['BASEFORM'] if len(a_t)>0 else t
            lemmas.append(lemma.lower())
    return " ".join(lemmas)

def lemmatize(inputstr, lemmatizer="tnpp"):
    if lemmatizer == "tnpp":
        return lemmatize_using_tnpp(inputstr)
    elif lemmatizer == "voikko":
        return lemmatize_using_voikko(inputstr)
    elif lemmatizer == "snowball":
        assert 0, "not yet implemented"
    else:
        assert 0, "unsupported lemmatizer"

def test_lemmatizer(lemmatizer="tnpp"):
    assert lemmatize("Testiä", lemmatizer) == "testi"
    return None

if __name__ == "__main__":
    teststr = "Tämä on testilause, joka nyt lemmatisoidaan."
    print(teststr, "=>", lemmatize(teststr))
