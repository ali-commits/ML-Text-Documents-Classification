from cleaner import stringClean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
from get_data import getData
import os

if(not os.path.isfile("../csv/dataset.csv")):
    getData()


data = pd.read_csv('../csv/dataset.csv')
x = data['document'].tolist()
y = data['type'].tolist()

for index, value in enumerate(x):
    print("processing data:", index, end="\r")
    x[index] = ' '.join([word for word in stringClean(value).split()])
print("processing data:", len(x))
vect = TfidfVectorizer(stop_words='english', min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)

print("no of features extracted:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)

pd.DataFrame(X_train).to_csv("../csv/train.csv")
pd.DataFrame(X_test).to_csv("../csv/test.csv")
pd.DataFrame(y_train).to_csv("../csv/train_lables.csv")
pd.DataFrame(y_test).to_csv("../csv/test_lables.csv")

print("train size:", X_train.shape)
print("test size:", X_test.shape)

model = SVC(gamma='scale', kernel='rbf', probability=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:  ", round(acc*100, 2), "%", sep="")

with open("../bin/model.bin", "wb") as modelFile:
    pickle.dump(model, modelFile)

with open("../bin/featuresVictor.bin", "wb") as featuresFile:
    pickle.dump(vect, featuresFile)
