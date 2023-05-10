import pandas as pd
import numpy as np
import os

import re
import torch
from torch.utils.data import Dataset
import pickle
from sklearn.model_selection import train_test_split

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# dealing with words
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))


class FncDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        # Assuming the target variable is in the last column
        features = sample[:-1]
        target = sample[-1]
        return torch.Tensor(features), torch.Tensor([target])


class DSLoader:
    def __init__(self, ds_path, load=True, balanceRelated=True):
        self.ds = self.getDS(ds_path)
        self.lemmatizerator = WordNetLemmatizer()
        self.c = 0
        self.g = 0

        if load and self.loadCheck():
            self.load()
        else:
            self.cleanDS()

        if balanceRelated:
            self.balanceRelated()

    def loadCheck(self):
        if os.path.exists("data/ds.pkl"):
            return True
        else:
            return False

    def load(self):
        with open("data/ds.pkl", "rb") as file:
            self.ds = pickle.load(file)

    def getDS(self, path):
        base_path = os.path.join(os.getcwd(), path)
        bodies_path = os.path.join(base_path, "train_bodies.csv")
        stances_path = os.path.join(base_path, "train_stances.csv")

        bodies = pd.read_csv(bodies_path)
        stances = pd.read_csv(stances_path)

        return pd.merge(bodies, stances, on="Body ID")

    def removeURL(self, s):
        """
        modified from [https://stackoverflow.com/a/11332580/8061549]
        """
        return re.sub(r'https?://\S+|www\.\S+', '', s, flags=re.MULTILINE)

    def removeHTML(self, s):
        return re.sub('<.*?>', '', s, flags=re.MULTILINE)

    def removeEmoji(self, s):
        """
        modified from [https://stackoverflow.com/a/33417311/8061549]
        """
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', s)

    def lemmatization(self, s):
        print(f"{(self.c / len(self.ds)) * 100}%")
        self.c += 1

        pattern = re.compile(r'\s')
        return ' '.join(self.lemmatizerator.lemmatize(w) for w in s.split(" "))

    def removeStopWords(self, s):
        return ' '.join(w for w in s.split(" ") if w not in stop_words)

    def filterLetters(self, s):
        return re.sub(r'[^a-zA-Z\s]', '', s).replace('\n', '').replace('\r', '').lower()


    def cleanDS(self):
        for coll_name in ["articleBody", "Headline"]:
            for f in [self.removeURL, self.removeHTML, self.removeEmoji, self.lemmatization, self.removeStopWords, self.filterLetters]:
                self.ds[coll_name] = self.ds[coll_name].apply(f)

        with open("data/ds.pkl", "wb") as file:
            pickle.dump(self.ds, file)

    def balanceRelated(self):
        min_count = sum(self.ds["Stance"].value_counts()) - self.ds["Stance"].value_counts()["unrelated"]
        self.ds = self.ds.groupby('Stance').apply(lambda x: x.sample(min(len(x), min_count))).reset_index(drop=True)


def split(ds, test=0.2, val=0.1):
    train = 1 - test - val
    X_train, X_test, y_train, y_test = train_test_split(ds[["articleBody", "Headline"]], ds["Stance"], test_size=test,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=val / (train + val),
                                                      random_state=1)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


if __name__ == '__main__':
    ds_path = "fnc-1"
    ds = DSLoader(ds_path, load=False)
    ds.balanceRelated()
    split(ds.ds, test=0.2, val=0.1)
