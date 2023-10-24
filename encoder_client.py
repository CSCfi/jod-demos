import requests
import numpy as np

def return_encodings(txt, verbose=False):
    if txt is None or len(txt)<4:
        return None, None

    try:
        r = requests.get('http://localhost:12123', params={'text': txt})
    except:
        assert 0, 'Connection to encoder-server not working. Is the server running?'
    assert r.status_code == requests.codes.ok, "Status code not ok"

    if verbose:
        print(r.url)
        print(r.text)

    encodings = r.json()['encodings']
    if verbose:
        print(encodings)

    def toarray(etype):
        if etype in encodings:
            res = np.array(encodings[etype])
            if verbose:
                print(res)
        else:
            res = None
        return res
        
    return toarray('query_ft'), toarray('query_strans')

if __name__ == "__main__":
    return_encodings("kekkonen kasvaa pienempi", verbose=True)
