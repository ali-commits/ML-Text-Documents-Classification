import os
import pandas as pd
from sklearn.model_selection import train_test_split


folders = ["business", "entertainment", "politics", "sport", "tech"]


def getData(path="../dataset/"):

    os.chdir(path)
    folders = os.listdir()
    x = []
    y = []

    labels = {"labels": folders}
    df = pd.DataFrame(labels)
    print('writing csv flie ...')
    df.to_csv('../csv/labels.csv')

    for i in folders:
        files = os.listdir(i)
        for text_file in files:
            file_path = i + "/" + text_file
            print("reading file:", file_path)
            with open(file_path) as f:
                data = f.readlines()
            data = ' '.join(data)
            x.append(data)
            y.append(i)

    data = {'document': x, 'type': y}
    df = pd.DataFrame(data)
    print('writing csv flie ...')
    df.to_csv('../csv/dataset.csv')


if __name__ == "__main__":
    getData()
