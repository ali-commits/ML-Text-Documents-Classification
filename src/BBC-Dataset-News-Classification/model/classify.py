from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC
import pickle

with open("model.bin", "rb") as modelFile:
    model = pickle.load(modelFile)

Y = np.array(['sport'])


news = """It's what representing their country is all about -- walking out in New Zealand's All Blacks jersey, facing their opposition, and delivering a spine-tingling, hair-raising Haka before the whistle blows for kick-off.

The sights and sounds of the Haka -- feet stomping, fists pumping, vocal chords straining -- are deeply entrenched within New Zealand culture.
"For me, the Haka is a symbol of who we are and where we come from," former All Blacks captain Richie McCaw told CNN in 2015.
"This is who we are. Obviously it comes from a Maori background but I think it also resonates with all Kiwis """
x = ' '.join([word for word in stringClean(news).split()])
X = vect.transform([x])
# print(model.predict(X))
lables = ['business', 'entertainment', 'politics', 'sport', 'tech']

prob = model.predict_proba(X)
for value, lable in zip(prob[0], lables):
    print(lable, "\t\t", round(value*100, 2), "%", sep="")
