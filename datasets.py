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

        return features, target


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


def getdatasets(model, load):
    if model == "bert":
        embedding_type = BERTEmbedding
    else:
        embedding_type = TfIdfEmbedding

    if not load:
        dataset = DSLoader(ds_path, model="bert", load=True, balanceRelated=True, shuffle=True)
        embedding_obj = embedding_type(titles=dataset.titles, bodies=dataset.bodies,
                                       save_path=f"data/{model}/ds_encoded.pkl", load=False)

    else:
        embedding_obj = embedding_type(titles=None, bodies=None,
                                       save_path=f"data/{model}/ds_encoded.pkl", load=load)

    return embedding_obj.ds


def getDataLoaders(model, train=0.2, test=0.5, batch_sieze=32, load=True, dataloader=True):
    whole_ds = getdatasets(model, load)
    whole_ds = balanceRelated(whole_ds)
    whole_ds = shuffle(whole_ds)

    train, test, val = split(whole_ds, test=0.2, val=0.5)
    train, test, val = EncodingDataset(train), EncodingDataset(test), EncodingDataset(val)

    if dataloader:
        train = torch.utils.data.DataLoader(train, batch_size=batch_sieze, shuffle=True)
        test = torch.utils.data.DataLoader(test, batch_size=batch_sieze, shuffle=True)
        val = torch.utils.data.DataLoader(val, batch_size=batch_sieze, shuffle=True)

    return train, test, val


if __name__ == '__main__':
    getDataLoaders(model="bert", load=True)
