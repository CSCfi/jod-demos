import requests

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

def lemmatize(inputstr):
        return lemmatize_using_tnpp(inputstr)

def test_lemmatizer():
    assert lemmatize("Testiä") == "testi"
    return None

if __name__ == "__main__":
    teststr = "Tämä on testilause, joka nyt lemmatisoidaan."
    print(teststr, "=>", lemmatize(teststr))
