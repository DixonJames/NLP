import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import *


class TfIdfEmbedding:
    def __init__(self, dataset):
        self.ds = dataset
        self.vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2)

        self.word_embedding_matrix = None
        self.document_embedding_matrix = None
        self.vocabulary = None

    def fit(self):
        all_passages = list(self.ds["articleBody"].values)
        all_passages.extend(list(self.ds["Headline"].values))

        self.word_embedding_matrix = self.vectorizer.fit_transform(all_passages)
        self.vocabulary = self.vectorizer.get_feature_names_out()
        self.document_embedding_matrix = self.word_embedding_matrix.toarray()

    def transform(self):
        self.ds


class BERTEmbeding:
    def __init__(self, dataset):
        self.DS = dataset


if __name__ == '__main__':


    ds_path = "fnc-1"
    ds = DSLoader(ds_path, load=True).ds

    tfidf = TfIdfEmbedding(ds)
    tfidf.fit()
    tfidf.transform()
