import numpy as np

from encoding import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

SEED = 42

Stance_conversion = {"unrelated": 0, "agree": 1, "discuss": 3, "disagree": 2}


class EncodingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        features = np.array(sample[["articleBody", "Headline"]])
        target = np.array(Stance_conversion[sample['Stance']])

        return np.array((features[0].flatten(), features[1].flatten())), target.tolist()

class EncodingDatasets(Dataset):
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def __len__(self):
        return len(self.df1)

    def __getitem__(self, index):
        sample1 = self.df1.iloc[index]
        features1 = np.array(sample1[["articleBody", "Headline"]])
        target1 = np.array(Stance_conversion[sample1['Stance']])

        sample2 = self.df1.iloc[index]
        features2 = np.array(sample2[["articleBody", "Headline"]])


        return np.array((features1[0].flatten(), features1[1].flatten())), np.array((features2[0].flatten(), features2[1].flatten())), target1.tolist()


def balanceRelated(ds):
    min_count = sum(ds["Stance"].value_counts()) - ds["Stance"].value_counts()["unrelated"]
    ds = ds.groupby('Stance').apply(lambda x: x.sample(min(len(x), min_count))).reset_index(drop=True)
    return ds

def balanceAllCat(ds):
    min_count = sum(ds["Stance"].value_counts()) - ds["Stance"].value_counts()["unrelated"]
    ds = ds.groupby('Stance').apply(lambda x: x.sample(min(len(x), min_count))).reset_index(drop=True)
    return ds


def shuffle(ds):
    random_order = np.random.permutation(ds.index)
    ds = ds.loc[random_order]
    ds = ds.reset_index(drop=True)
    return ds


def split(ds, test=0.2, val=0.5):
    train, test = train_test_split(ds[["articleBody", "Headline", "Stance"]], test_size=test,
                                   random_state=SEED)
    test, val = train_test_split(test, test_size=val,
                                 random_state=SEED)

    return train, test, val


def getdatasets(model, load, embed=True):
    ds_path = f"fnc-1"
    if model == "bert":
        embedding_type = BERTEmbedding
    else:
        embedding_type = TfIdfEmbedding

    if not load:

        dataset = DSLoader(ds_path, model=model, load=True, balanceRelated=True, shuffle=True)
        embedding_obj = embedding_type(titles=dataset.titles, bodies=dataset.bodies,
                                       save_path=f"data/{model}/ds_encoded.pkl", load=False, embed=embed)

    else:
        dataset = DSLoader(ds_path, model=model, load=True, balanceRelated=True, shuffle=load)
        embedding_obj = embedding_type(titles=dataset.titles, bodies=dataset.bodies,
                                       save_path=f"data/{model}/ds_encoded.pkl", load=load, embed=embed)

    return embedding_obj.ds

def getBothdatasets(load, embed=True):
    ds_path = f"fnc-1"

    dataset1 = DSLoader(ds_path, model="bert", load=True, balanceRelated=True, shuffle=True)
    embedding_obj1 = BERTEmbedding(titles=dataset1.titles, bodies=dataset1.bodies,
                                   save_path=f"data/bert/ds_encoded.pkl", load=False, embed=embed)


    dataset2 = DSLoader(ds_path, model="tfidf", load=True, balanceRelated=True, shuffle=load)
    embedding_obj2 = TfIdfEmbedding(titles=dataset2.titles, bodies=dataset2.bodies,
                                   save_path=f"data/tfidf/ds_encoded.pkl", load=load, embed=embed)

    return embedding_obj1.ds, embedding_obj2.ds


def getDataLoaders(model, train=0.2, test=0.5, batch_sieze=32, load=True, dataloader=True, embed=True, shuffle_ds=True, balance=True, balanceAll=False):
    """
    get; balance for relatedness; shffule; and plits the dataset
    :param model: tfidf or bert
    :param train:
    :param test:
    :param batch_sieze:
    :param load: wither to lad from pickle
    :param dataloader: to get it as an iterable in batchsize
    :param embed: if true will run though bert model, otherwise will get back a dict of parts ready to go thoguh model
    :return:
    """
    whole_ds = getdatasets(model, load)
    if balance:
        whole_ds = balanceRelated(whole_ds)
    if balanceAll:
        whole_ds = balanceAllCat(whole_ds)
    if shuffle_ds:
        whole_ds = shuffle(whole_ds)

    train, test, val = split(whole_ds, test=0.2, val=0.5)
    train, test, val = EncodingDataset(train), EncodingDataset(test), EncodingDataset(val)
    # train, test, val = train.df, test.df, val.df
    if dataloader:
        train = torch.utils.data.DataLoader(train, batch_size=batch_sieze, shuffle=shuffle_ds)
        test = torch.utils.data.DataLoader(test, batch_size=batch_sieze, shuffle=shuffle_ds)
        val = torch.utils.data.DataLoader(val, batch_size=batch_sieze, shuffle=shuffle_ds)

    return train, test, val

def getCombinedDataLoaders(model, train=0.2, test=0.5, batch_sieze=32, load=True, dataloader=True, embed=True, shuffle_ds=True, balance=True, balanceAll=False):
    """
    get; balance for relatedness; shffule; and plits the dataset
    :param model: tfidf or bert
    :param train:
    :param test:
    :param batch_sieze:
    :param load: wither to lad from pickle
    :param dataloader: to get it as an iterable in batchsize
    :param embed: if true will run though bert model, otherwise will get back a dict of parts ready to go thoguh model
    :return:
    """
    whole_ds1, whole_ds2 = getBothdatasets(load)
    if balance:
        whole_ds1 = balanceRelated(whole_ds1)
        whole_ds2 = balanceRelated(whole_ds2)
    if balanceAll:
        whole_ds1 = balanceAllCat(whole_ds1)
        whole_ds2 = balanceAllCat(whole_ds2)
    if shuffle_ds:
        whole_ds1 = shuffle(whole_ds1)
        whole_ds2 = shuffle(whole_ds2)

    train1, test1, val1 = split(whole_ds1, test=0.2, val=0.5)
    train2, test2, val2 = split(whole_ds2, test=0.2, val=0.5)
    train, test, val = EncodingDatasets(train1, train2), EncodingDatasets(test1, test2), EncodingDatasets(val1, val2)

    if dataloader:
        train = torch.utils.data.DataLoader(train, batch_size=batch_sieze, shuffle=shuffle_ds)
        test = torch.utils.data.DataLoader(test, batch_size=batch_sieze, shuffle=shuffle_ds)
        val = torch.utils.data.DataLoader(val, batch_size=batch_sieze, shuffle=shuffle_ds)

    return train, test, val


def getRelatedDataLoaders(model, train=0.2, test=0.5, batch_sieze=32, load=True, dataloader=True, embed=True):
    """
    get; balance for relatedness; shffule; and plits the dataset
    :param model: tfidf or bert
    :param train:
    :param test:
    :param batch_sieze:
    :param load: wither to lad from pickle
    :param dataloader: to get it as an iterable in batchsize
    :param embed: if true will run though bert model, otherwise will get back a dict of parts ready to go thoguh model
    :return:
    """
    whole_ds = getdatasets(model, load)
    whole_ds = balanceRelated(whole_ds)
    whole_ds = shuffle(whole_ds)
    whole_ds = whole_ds[whole_ds['Stance'] != 'unrelated']

    train, test, val = split(whole_ds, test=0.2, val=0.5)
    train, test, val = EncodingDataset(train), EncodingDataset(test), EncodingDataset(val)
    # train, test, val = train.df, test.df, val.df
    if dataloader:
        train = torch.utils.data.DataLoader(train, batch_size=batch_sieze, shuffle=True)
        test = torch.utils.data.DataLoader(test, batch_size=batch_sieze, shuffle=True)
        val = torch.utils.data.DataLoader(val, batch_size=batch_sieze, shuffle=True)

    return train, test, val


if __name__ == '__main__':
    train, test, val = getDataLoaders(model="bert", load=True, embed=True)
    #[a for a in val]
    print("done")
