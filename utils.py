import pandas as pd
import numpy as np
import os
import random
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

random.seed(42)

sns.set_style("whitegrid")


class DSLoader:
    def __init__(self, ds_path, model, load=True, balanceRelated=True, shuffle=True):
        self.ds = None
        self.bodies = None
        self.titles = None
        self.lemmatizerator = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.model = model
        self.c = 0
        self.g = 0

        self.body_save_path = f"data/{model}/body_ds.pkl"
        self.title_save_path = f"data/{model}/title_ds.pkl"

        self.getDS(ds_path)

        if load and self.loadCheck():
            self.load()
        else:
            self.cleanDS()

    def loadCheck(self):
        if os.path.exists(self.body_save_path) and os.path.exists(self.title_save_path):
            return True
        else:
            return False

    def load(self):
        with open(self.body_save_path, "rb") as file:
            self.bodies = pickle.load(file)

        with open(self.title_save_path, "rb") as file:
            self.titles = pickle.load(file)

    def save(self):
        with open(self.body_save_path, "wb") as file:
            pickle.dump(self.bodies, file)

        with open(self.title_save_path, "wb") as file:
            pickle.dump(self.titles, file)

    def getDS(self, path):
        base_path = os.path.join(os.getcwd(), path)
        bodies_path = os.path.join(base_path, "train_bodies.csv")
        stances_path = os.path.join(base_path, "train_stances.csv")

        self.bodies = pd.read_csv(bodies_path)
        self.titles = pd.read_csv(stances_path)

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
        return ' '.join(self.lemmatizerator.lemmatize(w) for w in s.split(" "))

    def stem(self, s):
        return ' '.join(self.stemmer.stem(w) for w in s.split(" "))

    def removeStopWords(self, s):
        return ' '.join(w for w in s.split(" ") if w not in stop_words)

    def filterLetters(self, s):
        return re.sub(r'[^a-zA-Z\s.]', '', s).replace('\n', '').replace('\r', '').lower()

    def filterPeriods(self, s):
        s = s.replace('.', '')
        s = " ".join([" ".join(word_tokenize(s)) for s in s.split(" ") if s.isalpha() and s not in stop_words])
        return s

    def cleanDS(self):
        for df, coll in zip([self.bodies, self.titles], ["articleBody", "Headline"]):
            df_lst = [v.split(" ") for v in df[coll].values]
            clearned_df = []
            for row in df_lst:
                cleared_row = []
                for word in row:
                    word = word.lower()
                    if word not in stop_words and "http" not in word:
                        if self.model != "bert":
                            word = self.stemmer.stem(word)
                            word = self.lemmatizerator.lemmatize(word)
                        word = re.sub(r'[^a-zA-Z\s.]', '', word).replace('\n', '').replace('\r', '')

                        cleared_row.append(word)
                clearned_df.append(" ".join(cleared_row))
            df[coll] = pd.Series(clearned_df)

        # self.merge()
        self.save()


def split(ds, test=0.2, val=0.1):
    train = 1 - test - val
    X_train, X_test, y_train, y_test = train_test_split(ds[["articleBody", "Headline"]], ds["Stance"], test_size=test,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=val / (train + val),
                                                      random_state=1)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


class ResultsDisplay:
    def __init__(self, model, real_values, test_df, labels, title):
        self.model = model
        self.test_df = test_df

        self.real_values = real_values

        self.title = title
        self.labels = labels


        self.predicted_categories = None
        self.predicted_probs = None

        self.predict()

    def predict(self):
        test_x = self.test_df[0]
        # test_y = test_df[1]

        test_x = np.array([a for a in np.array(test_x.values)])
        # test_y = np.array([relatedness_conversion[int(a.max())] for a in np.array(test_y.values)])

        self.predicted_probs = self.model.predict_proba(test_x)[::,1]
        self.predicted_categories = (self.model.predict(test_x) >= 0.5).astype(int)

    def metrics(self):
        print(classification_report(self.real_values, self.predicted_categories))

    def confusionMatrix(self):
        """
        modified from [https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/]
        :return:
        """
        cm = pd.DataFrame(confusion_matrix(self.real_values, self.predicted_categories), columns=self.labels,
                          index=self.labels)

        sns.heatmap(cm,
                    annot=True, cmap="coolwarm")
        plt.ylabel('Prediction', fontsize=13)
        plt.xlabel('Actual', fontsize=13)
        plt.title(f'{self.title} - Confusion Matrix', fontsize=17)
        plt.show()

    def rocCurve(self):
        fpr, tpr, _ = roc_curve(self.real_values, self.predicted_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 7))
        plt.title(f'{self.title} - ROC Curve', fontsize=17)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.show()


def createCleanDS():
    """
    loads and saves datasets for BERT and tfidf
    these differ as some of the text clearning differs between them
    :return:
    """
    ds_path = "fnc-1"
    ds_bert = DSLoader(ds_path=ds_path, model="bert", load=False, balanceRelated=True, shuffle=True)
    ds_tfidf = DSLoader(ds_path=ds_path, model="tfidf", load=False, balanceRelated=True, shuffle=True)


if __name__ == '__main__':
    createCleanDS()
    # split(ds.ds, test=0.2, val=0.1)
