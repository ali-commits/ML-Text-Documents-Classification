from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
from cleaner import stringClean
from sklearn.svm import SVC


class classifier():

    def __init__(self, modelFile="../bin/model.bin", featuresFile="../bin/featuresVictor.bin"):
        with open(modelFile, "rb") as f:
            self.model = pickle.load(f)
        with open(featuresFile, "rb") as f:
            self.vect = pickle.load(f)

        self.lables = pd.read_csv('../csv/labels.csv')['labels'].tolist()

    def classify(self, news=''):
        x = ' '.join([word for word in stringClean(news).split()])
        X = self.vect.transform([x])
        prob = self.model.predict_proba(X)
        prob = list(prob[0])
        for i in range(len(prob)):
            prob[i] = round(prob[i]*100, 2)
        results = dict(zip(self.lables, prob))
        results = sorted(results.items(), key=lambda kv: (
            kv[1], kv[0]), reverse=True)
        return results


if __name__ == "__main__":

    news = """It's what representing their country is all about -- walking out in New Zealand's All Blacks jersey, facing their opposition, and delivering a spine-tingling, hair-raising Haka before the whistle blows for kick-off.
    The sights and sounds of the Haka -- feet stomping, fists pumping, vocal chords straining -- are deeply entrenched within New Zealand culture.
    "For me, the Haka is a symbol of who we are and where we come from," former All Blacks captain Richie McCaw told CNN in 2015.
    "This is who we are. Obviously it comes from a Maori background but I think it also resonates with all Kiwis """

    model = classifier()
    results = model.classify(news=news)
    for cat, value in results:
        print(cat, value)
