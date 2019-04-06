from cleaner import stringClean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import pickle


data = pd.read_csv('../dataset/dataset.csv')
x = data['news'].tolist()
y = data['type'].tolist()

for index, value in enumerate(x):
    print("processing data:", index, end="\r")
    x[index] = ' '.join([word for word in stringClean(value).split()])
print("processing data:", len(x))
vect = TfidfVectorizer(stop_words='english', min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)

print(X.shape)

print("no of features extracted:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)

print("train size:", X_train.shape)
print("test size:", X_test.shape)

model = SVC(gamma='scale', kernel='rbf', probability=True)


model.fit(X_train, y_train)

with open("model.bin", "wb") as modelFile:
    pickle.dump(model, modelFile)

with open("model.bin", "rb") as modelFile:
    model = pickle.load(modelFile)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:  ", round(acc*100, 2), "%", sep="")


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
