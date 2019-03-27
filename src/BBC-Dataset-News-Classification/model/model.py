import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC
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

model = SVC(gamma='scale', kernel='rbf')
model.fit(X_train, y_train)

# with open("model.bin", "wb") as modelFile:
#     pickle.dump(model, modelFile)

# with open("model.bin", "rb") as modelFile:
#     model = pickle.load(modelFile)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("\nAccuracy:  ", round(acc*100,2), "%", sep="")
print(X_test.shape)


# Y = np.array(['sport'])    


# news = """but the match was overshadowed by racist chanting from some home fans directed at several England players, including Danny Rose.

# Uefa said "disciplinary proceedings" had been opened against Montenegro with one charge for "racist behaviour".

# The case will be dealt with by European football's governing body on 16 May.

# Montenegro coach Ljubisa Tumbakovic said he did not "hear or notice any" racist abuse.

# But England manager Gareth Southgate, speaking to BBC Radio 5 Live said he "definitely heard the racist abuse of Rose".

# "There's no doubt in my mind it happened," he added. "I know what I heard. It's unacceptable.

# "We have to make sure our players feel supported, they know the dressing room is there and we as a group of staff are there for them.

# "We have to report it through the correct channels. It is clear that so many people have heard it and we have to continue to make strides in our country and trust the authorities to take the right action."

# Anti-discrimination group Fare said they had identified the match as "high risk" for racism before the game and executive director Piara Powar said: "We had an observer present who picked up evidence of racial abuse.

# "Our monitoring team have been compiling the evidence we have before presenting it to Uefa."

# Montenegro also face other charges relating to crowd disturbances, the throwing of objects, setting off of fireworks and the blocking of stairways following the game at the Podgorica City Stadium.

# The minimum punishment from Uefa for an incident of racism is a partial stadium closure, while a second offence results in one match being played behind closed doors and a fine of 50,000 euros (£42,500).

# Uefa rules add: "Any subsequent offence is punished with more than one match behind closed doors, a stadium closure, the forfeiting of a match, the deduction of points and/or disqualification from the competition."""
# x = ' '.join([word for word in stringClean(news).split()])
# X = vect.fit([x])
# print(X.shape)

