import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
from utils import *


class TfIdfEmbedding:
    def __init__(self, dataset, load=False):
        self.ds = dataset
        self.vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2)
        self.load = load

        self.word_embedding_matrix = None
        self.document_embedding_matrix = None
        self.vocabulary = None

        self.clean()
        self.transform()

    def clean(self):
        def filterPeriods(s):
            return s.replace('.', '')

        for coll_name in ["articleBody", "Headline"]:
            self.ds[coll_name] = self.ds[coll_name].apply(filterPeriods)

    def transform(self):
        if not self.load:
            body = list(self.ds["articleBody"].values.astype('U'))
            head = list(self.ds["Headline"].values.astype('U'))

            head_embedding_matrix = self.vectorizer.fit_transform(head)
            head_document_embedding_matrix = head_embedding_matrix.toarray()

            body_embedding_matrix = self.vectorizer.fit_transform(head)
            body_document_embedding_matrix = body_embedding_matrix.toarray()

            vocabulary = self.vectorizer.get_feature_names_out()

        else:
            with open("data/tfidf/head_embedding.pkl", "rb") as file:
                head_document_embedding_matrix = pickle.load(file)

            with open("data/tfidf/body_embedding.pkl", "rb") as file:
                body_document_embedding_matrix = pickle.load(file)

        self.ds["articleBody"] = pd.Series(
            list(body_document_embedding_matrix[i]) for i in range(len(body_document_embedding_matrix)))
        self.ds["Headline"] = pd.Series(
            list(head_document_embedding_matrix[i]) for i in range(len(head_document_embedding_matrix)))


class BERTEmbeding:
    def __init__(self, dataset):
        self.ds = dataset
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(self.model_name)

    def tokensise(self):
        for coll_name in ["articleBody", "Headline"]:
            passages = self.ds[coll_name].values.tolist()
            encoded_inputs = [self.tokenizer.encode_plus(passage,
                                                         add_special_tokens=True,
                                                         padding=True,
                                                         truncation=True,
                                                         return_tensors='pt') for passage in passages]
            input_ids = encoded_inputs['input_ids']
            attention_mask = encoded_inputs['attention_mask']
            token_type_ids = encoded_inputs['token_type_ids']

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                word_embeddings = outputs.last_hidden_state
                passage_embeddings = outputs.pooler_output


if __name__ == '__main__':
    ds_path = "fnc-1"
    ds = DSLoader(ds_path, load=True).ds

    #tfidf = TfIdfEmbedding(ds, load=True)
    bert = BERTEmbeding(ds)
    bert.tokensise()
