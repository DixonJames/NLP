import numpy as np

from encoding import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

SEED = 42

Stance_conversion = {"unrelated": 0, "agree":1, "discuss":3, "disagree":2}


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


def balanceRelated(ds):
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


def getDataLoaders(model, train=0.2, test=0.5, batch_sieze=32, load=True, dataloader=True, embed=True):
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

    train, test, val = split(whole_ds, test=0.2, val=0.5)
    train, test, val = EncodingDataset(train), EncodingDataset(test), EncodingDataset(val)
    #train, test, val = train.df, test.df, val.df
    if dataloader:
        train = torch.utils.data.DataLoader(train, batch_size=batch_sieze, shuffle=True)
        test = torch.utils.data.DataLoader(test, batch_size=batch_sieze, shuffle=True)
        val = torch.utils.data.DataLoader(val, batch_size=batch_sieze, shuffle=True)

    return train, test, val


if __name__ == '__main__':
    train, test, val = getDataLoaders(model="bert", load=True, embed=True)
    [a for a in val]
    print("done")
