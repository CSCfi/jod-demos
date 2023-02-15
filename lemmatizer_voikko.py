from nltk.tokenize import word_tokenize
import libvoikko
v = libvoikko.Voikko(u"fi")

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

def lemmatize(inputstr):
    return lemmatize_using_voikko(inputstr)

def test_lemmatizer():
    assert lemmatize("Testiä") == "testi"
    return None

if __name__ == "__main__":
    teststr = "Tämä on testilause, joka nyt lemmatisoidaan."
    print(teststr, "=>", lemmatize(teststr))
