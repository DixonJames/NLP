from encoding import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

SEED = 42

relevance_conversion = {"related": 1, "unrelated": 0}


class EncodingDataset(Dataset):
    def __init__(self, ds_generator):
        self.ds_generator = ds_generator

    def __len__(self):
        return len(self.ds_generator)

    def __getitem__(self, item):
        row = self.ds_generator.getItem(item)
        row["Stance"] = relevance_conversion[row["Stance"]]

        return row


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


def getDataLoaders(model, load=True):
    ds_path = "fnc-1"
    dataset = DSLoader(ds_path, model=model, load=load, balanceRelated=True, shuffle=True)
    ds = dataset.ds

    train, test, val = split(ds)

    if model == "bert":

        train_dataGenerator = BERTEmbedding(train, save_path=f"data/{model}/train.pkl", load=False)
        test_dataGenerator = BERTEmbedding(test, save_path=f"data/{model}/test.pkl", load=False)
        val_dataGenerator = BERTEmbedding(val, save_path=f"data/{model}/val.pkl", load=False)

    elif model == "tfidf":
        train_dataGenerator = TfIdfEmbedding(ds, save_path=f"data/{model}/ds_encoded.pkl", load=False)

    print("done")


if __name__ == '__main__':
    getDataLoaders(model="tfidf")
