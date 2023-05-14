import pickle
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import torch
from utils import *
from nltk.tokenize import word_tokenize
import seaborn as sns
import math
import matplotlib.pyplot as plt
from collections import defaultdict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
stop_words = set(stopwords.words('english'))


class TfIdfEmbedding:
    """
    modified from [https://www.askpython.com/python/examples/tf-idf-model-from-scratch]
    """

    def __init__(self, titles, bodies, save_path, load=False, embed=None):
        self.ds = None
        self.titles = titles
        self.bodies = bodies

        self.save_path = save_path
        self.head_vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2, stop_words='english')
        self.body_vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2)
        self.load_bool = load

        self.word_embedding_matrix = None
        self.document_embedding_matrix = None
        self.vocabulary = None

        self.counter = 0

        self.word_set = set()
        self.word_count = defaultdict(int)
        self.index_pos = dict()

        if not load:
            self.clean()
            self.createSetWords()
            self.transform()
        else:
            self.load()

    def save(self):
        with open(self.save_path, "wb") as file:
            pickle.dump(self.ds, file)

    def load(self):
        with open(self.save_path, "rb") as file:
            self.ds = pickle.load(file)

    def merge(self):
        self.ds = pd.merge(self.bodies, self.titles, on="Body ID")

    def clean(self):
        def filterPeriods(s):
            # print(f"{(self.counter / len(self.ds)) * 100}")
            self.counter += 1
            s = s.replace('.', '')
            s = " ".join([s for s in s.split(" ") if s.isalpha() and s not in stop_words])
            return s

        for df, coll in zip([self.bodies, self.titles], ["articleBody", "Headline"]):
            df[coll] = df[coll].apply(filterPeriods)

    def batch_iterator(self, strings, batch_size=1000):
        for i in range(0, len(strings), batch_size):
            yield strings[i:i + batch_size]

    def createSetWords(self):
        for df, coll in zip([self.bodies, self.titles], ["articleBody", "Headline"]):
            for doc in df[coll]:
                self.word_set.update(set([w for w in doc.split(" ")]))
                for w in doc.split(" "):
                    self.word_count[w] += 1

        i = 0
        for word in self.word_set:
            if self.word_count[word] > 5:
                self.index_pos[word] = i
                i += 1
        self.word_set = set([k for k in self.word_count.keys() if self.word_count[k] > 5])

    def tf(self, docs):

        def default_int_dict():
            return defaultdict(int)

        tf = defaultdict(default_int_dict)

        i = 0
        for doc in docs:
            for token in doc.split(" "):
                if token in self.word_set:
                    tf[str(i)][token] += 1
            i += 1
            # print(f"tf: {i / len(docs) * 100}%")
        return tf

    def idf(self, docs):
        idf = defaultdict(int)

        i = 0
        for doc in docs:
            # print(f"idf: {i / len(docs) * 100}%")
            i += 1

            for token in doc.split(" "):
                if token in self.word_set:
                    idf[token] += 1

        for token, count in idf.items():
            idf[token] = math.log(len(docs) / count)

        return idf

    def tfidf(self, tf, idf, doc, row_num):
        tf_idf_vec = np.zeros((len(self.word_set),))
        for word in doc.split(" "):
            if word in self.word_set:
                tf_idf_vec[self.index_pos[word]] = tf[str(row_num)][word] * idf[word]
        return tf_idf_vec

    def transform(self):
        for df, coll in zip([self.bodies, self.titles], ["articleBody", "Headline"]):
            docs = df[coll]
            tf = self.tf(docs)
            idf = self.idf(docs)

            df[coll] = [self.tfidf(tf, idf, row, index) for index, row in enumerate(docs)]

        self.merge()
        self.save()

    def __len__(self):
        return len(self.ds)

    def recordGen(self):
        for index, row in self.ds.iterrows():
            yield {"Headline": row["Headline"],
                   "articleBody": row["articleBody"],
                   "stance": row["Stance"]}

    def getItem(self, i):
        row = self.ds.iloc[i]
        yield {"Headline": row["Headline"],
               "articleBody": row["articleBody"],
               "stance": row["Stance"]}


class BERTEmbedding:
    def __init__(self, titles, bodies, save_path, load=False, embed=True):
        self.ds = None
        self.titles = titles
        self.bodies = bodies

        self.model_name = 'bert-base-uncased'
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(self.model_name)

        self.body_length = 64
        self.head_length = 32
        self.counter = 0

        if load:
            self.load()
        else:
            self.tokensise(embed)

    def merge(self):
        self.ds = pd.merge(self.bodies, self.titles, on="Body ID")

    def save(self):
        with open(self.save_path, "wb") as file:
            pickle.dump(self.ds, file)

    def load(self):
        with open(self.save_path, "rb") as file:
            self.ds = pickle.load(file)

    def shuffle(self):
        random_order = np.random.permutation(self.ds.index)
        self.ds = self.ds.loc[random_order]
        self.ds = self.ds.reset_index(drop=True)

    def checkTokenLength(self):

        for coll_name in ["articleBody", "Headline"]:
            token_lens = []
            for txt in self.ds[coll_name].values:
                tokens = self.tokenizer.encode(txt, max_length=1000)
                token_lens.append(len(tokens))

            if coll_name == "articleBody":
                sns.distplot(token_lens)
                plt.xlim([0, 1000])
                plt.savefig(coll_name + ".png")
            else:
                sns.distplot(token_lens)
                plt.xlim([0, 50])
                plt.savefig(coll_name + ".png")

    def tokensise(self, embed):
        """
        modified from [https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/]
        tokenizes the inputs; adds special charactors; converts to numerical representation
        :return:
        """

        def berttokenise(pasage, coll_name):
            if coll_name == "articleBody":
                length = self.body_length
            elif coll_name == "Headline":
                length = self.head_length

            # print(f"{(self.counter / len(self.ds)) * 100}%")
            # self.counter += 1
            return self.tokenizer.encode_plus(
                text=pasage,
                add_special_tokens=True,
                max_length=length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

        for df, coll_name in zip([self.bodies, self.titles], ["articleBody", "Headline"]):
            df[coll_name] = df[coll_name].apply(lambda passage: berttokenise(passage, coll_name))
            bert_encodings = []
            if embed:
                for index, row in df.iterrows():
                    print(f"{(self.counter / len(df)) * 100}%")
                    self.counter += 1
                    val = row[coll_name]
                    bert_encodings.append(self.bertEncode(val))
                df[coll_name] = bert_encodings

            with open(f"{coll_name}.pkl", "wb") as file:
                pickle.dump(bert_encodings, file)

            """with open(f"{coll_name}.pkl", "rb") as file:
                df[coll_name] = pickle.load(file)"""

        self.merge()
        self.save()

    def bertEncode(self, encoded_input):
        """print(f"{(self.counter / len(self.ds)) * 100}%")
        self.counter += 1"""
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        token_type_ids = encoded_input['token_type_ids']

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            word_embedding = outputs.last_hidden_state
            passage_embedding = outputs.pooler_output

        return passage_embedding.numpy()

    def encode(self):

        for coll_name in ["Headline", "articleBody"]:
            encodings = []
            for val in self.ds[coll_name].values:
                encodings.append(self.bertEncode(val))
            self.counter = 0

        self.save()

    def __len__(self):
        return len(self.ds)

    def recordGen(self):
        for index, row in self.ds.iterrows():
            yield {"Headline": self.bertEncode(row["Headline"]),
                   "articleBody": self.bertEncode(row["articleBody"]),
                   "stance": row["Stance"]}

    def getItem(self, i):
        row = self.ds.iloc[i]
        yield {"Headline": self.bertEncode(row["Headline"]),
               "articleBody": self.bertEncode(row["articleBody"]),
               "stance": row["Stance"]}


def createEncodedDS(ds_path):
    """
    creates the full merged dataset for bert and tfidf, ready for training
    :return:
    """

    tfidf_dataset = DSLoader(ds_path, model="tfidf", load=True, balanceRelated=True, shuffle=True)

    tfidf = TfIdfEmbedding(titles=tfidf_dataset.titles, bodies=tfidf_dataset.bodies,
                           save_path="data/tfidf/ds_encoded.pkl", load=False)

    bert_dataset = DSLoader(ds_path, model="bert", load=True, balanceRelated=True, shuffle=True)
    bert = BERTEmbedding(titles=bert_dataset.titles, bodies=bert_dataset.bodies,
                         save_path="data/bert/ds_encoded.pkl", load=False)





if __name__ == '__main__':
    ds_path = "fnc-1"
    createEncodedDS(ds_path)
