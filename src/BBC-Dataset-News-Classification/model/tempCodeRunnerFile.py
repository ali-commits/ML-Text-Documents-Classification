with open("model.bin", "wb") as modelFile:
    pickle.dump(model, modelFile)

with open("model.bin", "rb") as modelFile:
    model = pickle.load(modelFile)