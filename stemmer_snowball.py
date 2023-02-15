from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("finnish")

def stem_using_snowball(inputstr):
    tokens = word_tokenize(inputstr)
    stems = []
    for t in tokens:
        if len(t)>1:
            stems.append(stemmer.stem(t))
    return " ".join(stems)

def lemmatize(inputstr):
    return stem_using_snowball(inputstr)

def test_lemmatizer():
    assert lemmatize("Testiä") == "test"
    return None

if __name__ == "__main__":
    teststr = "Tämä on testilause, joka nyt lemmatisoidaan."
    print(teststr, "=>", lemmatize(teststr))
