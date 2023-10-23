import requests
import numpy as np

r = requests.get('http://localhost:12123', params={'text': 'kekkonen kukat autolla'})
assert r.status_code == requests.codes.ok
print(r.url)
print(r.text)
rjson = r.json() 
print(rjson)
print(rjson['encodings'])
print(rjson['encodings']['random'])
print(np.array(rjson['encodings']['random']))


