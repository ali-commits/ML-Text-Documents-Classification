from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC
import pickle

with open("model.bin", "rb") as modelFile:
    model = pickle.load(modelFile)

Y = np.array(['sport'])    


news = """but the match was overshadowed by racist chanting from some home fans directed at several England players, including Danny Rose.

Uefa said "disciplinary proceedings" had been opened against Montenegro with one charge for "racist behaviour".

The case will be dealt with by European football's governing body on 16 May.

Montenegro coach Ljubisa Tumbakovic said he did not "hear or notice any" racist abuse.

But England manager Gareth Southgate, speaking to BBC Radio 5 Live said he "definitely heard the racist abuse of Rose".

"There's no doubt in my mind it happened," he added. "I know what I heard. It's unacceptable.

"We have to make sure our players feel supported, they know the dressing room is there and we as a group of staff are there for them.

"We have to report it through the correct channels. It is clear that so many people have heard it and we have to continue to make strides in our country and trust the authorities to take the right action."

Anti-discrimination group Fare said they had identified the match as "high risk" for racism before the game and executive director Piara Powar said: "We had an observer present who picked up evidence of racial abuse.

"Our monitoring team have been compiling the evidence we have before presenting it to Uefa."

Montenegro also face other charges relating to crowd disturbances, the throwing of objects, setting off of fireworks and the blocking of stairways following the game at the Podgorica City Stadium.

The minimum punishment from Uefa for an incident of racism is a partial stadium closure, while a second offence results in one match being played behind closed doors and a fine of 50,000 euros (£42,500).

Uefa rules add: "Any subsequent offence is punished with more than one match behind closed doors, a stadium closure, the forfeiting of a match, the deduction of points and/or disqualification from the competition."""
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform([news])
print(X.shape)
# y_pred = model.predict(X)

# print(y_pred)
