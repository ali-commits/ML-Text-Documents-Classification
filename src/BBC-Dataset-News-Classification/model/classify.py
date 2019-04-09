from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from cleaner import stringClean
from sklearn.svm import SVC

# """It's what representing their country is all about -- walking out in New Zealand's All Blacks jersey, facing their opposition, and delivering a spine-tingling, hair-raising Haka before the whistle blows for kick-off.
# The sights and sounds of the Haka -- feet stomping, fists pumping, vocal chords straining -- are deeply entrenched within New Zealand culture.
# "For me, the Haka is a symbol of who we are and where we come from," former All Blacks captain Richie McCaw told CNN in 2015.
# "This is who we are. Obviously it comes from a Maori background but I think it also resonates with all Kiwis """


class classifier():
    
    def __init__(self, modelFile="model.bin", featuresFile="tfidf.bin"):
        with open(modelFile, "rb") as f:
            self.model = pickle.load(f)
        with open(featuresFile, "rb") as f:
            self.vect = pickle.load(f)
        self.lables = ['business', 'entertainment',
                       'politics', 'sport', 'tech']

    def classify(self, news=''):
        x = ' '.join([word for word in stringClean(news).split()])
        X = self.vect.transform([x])
        prob = self.model.predict_proba(X)
        return dict(zip(self.lables, prob[0]))


if __name__ == "__main__":
    news = """It's what representing their country is all about -- walking out in New Zealand's All Blacks jersey, facing their opposition, and delivering a spine-tingling, hair-raising Haka before the whistle blows for kick-off.
    The sights and sounds of the Haka -- feet stomping, fists pumping, vocal chords straining -- are deeply entrenched within New Zealand culture.
    "For me, the Haka is a symbol of who we are and where we come from," former All Blacks captain Richie McCaw told CNN in 2015.
    "This is who we are. Obviously it comes from a Maori background but I think it also resonates with all Kiwis """

    model = classifier()
    preb = model.classify(news=news)
    print(sorted(preb.items(), key = 
             lambda kv:(kv[0], kv[1])))    
