import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import pickle


def stringClean(string):
   
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\.", "", string)
    return string.strip().lower()

data = pd.read_csv('../dataset/dataset.csv')
x = data['news'].tolist()
y = data['type'].tolist()

for index,value in enumerate(x):
    print( "processing data:", index, end="\r" )
    x[index] = ' '.join([word for word in stringClean(value).split()])
print( "processing data:",len(x))
vect = TfidfVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)

print(X.shape)

print("no of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

print("train size:", X_train.shape)
print("test size:", X_test.shape)

model = SVC(gamma='scale', kernel='rbf', probability=True)


model.fit(X_train, y_train)

# with open("model.bin", "wb") as modelFile:
#     pickle.dump(model, modelFile)

# with open("model.bin", "rb") as modelFile:
#     model = pickle.load(modelFile)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)

print("\nAccuracy:  ", round(acc*100,2), "%", sep="")




news = """What looks like a can't-miss concept -- the aging lawmen who hunted down Bonnie and Clyde -- yields a dutiful, uninspired movie in "The Highwaymen," pairing Kevin Costner and Woody Harrelson as the taciturn Texas Rangers called out of retirement, which roughly approximates what will likely be the film's target demo.

Despite the star power, "Highwaymen" rather simple-mindedly follows a familiar road map, nostalgically hearkening back to a day when nobody needed to worry about reading Miranda rights and a cop could say -- as Costner's Frank Hamer does -- "You know you're gonna have to put this man down."
That man would be Clyde Barrow, who with Bonnie Parker left a trail of bodies in their wake, while achieving Depression-era celebrity -- cold-blooded killers who were, it's noted, "more adored than movie stars."
Unlike the 1967 movie with Warren Beatty and Faye Dunaway (where did the time go?), the outlaws are essentially relegated to an off-screen afterthought, following an introductory sequence in which Bonnie helps break Clyde out of a Texas prison in 1934.
Texas' governor, Miriam "Ma" Ferguson (Kathy Bates, without much to do but snarl), agrees to enlist former rangers to undertake the manhunt. But even she sounds dismissive of the aging cowboys, referring to them as "a couple of has-been vaqueros."
Hamer is reluctant to reunite with Harrelson's Maney Gault, which, like almost everything about their interactions, plays as a tough-guy-movie cliché -- something like "Grumpy Old Lawmen." That includes the darker past they'd rather not discuss, concerns about growing older and a growing sense of conviction thanks to the collateral damage they encounter in the course of their pursuit.
Needless to say, Costner and Harrelson are well-suited to these 20th-century cowboy roles -- Costner has more than done his part to help keep the western alive -- and director John Lee Hancock (whose credentials include the 2004 version of "The Alamo") and writer John Fusco have some fun with the idea that these aging manhunters can't, say, outrun younger suspects the way they once might have."""


x = ' '.join([word for word in stringClean(news).split()])
X = vect.transform([x])
print(model.predict(X))
print(model.predict_proba(X))

