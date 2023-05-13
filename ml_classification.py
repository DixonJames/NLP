import pickle

# remove on non intell processors or if getting weird errors!!!
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import pandas as pd
from datasets import getDataLoaders
import numpy as np
from sklearn.decomposition import PCA

from utils import ResultsDisplay

relatedness_conversion = {0: 0, 1: 1, 3: 1, 2: 1}


# pca = PCA(n_components=0.95)

def ML_train(train, limmit=0):
    train_x = train[0]
    train_y = train[1]

    train_x = np.array([a for a in np.array(train_x.values)])
    train_y = np.array([relatedness_conversion[int(a.max())] for a in np.array(train_y.values)])

    if limmit!=0:
        train_x = train_x[:limmit]
        train_y = train_y[:limmit]

    linear_regression = LinearRegression()

    linear_regression.fit(train_x, train_y)

    return linear_regression


def dfList(train, test, val):
    train = pd.DataFrame([(np.concatenate((v[0][0], v[0][1]), axis=0).flatten(), v[1]) for v in train])
    val = pd.DataFrame([(np.concatenate((v[0][0], v[0][1]), axis=0).flatten(), v[1]) for v in val])
    test = pd.DataFrame([(np.concatenate((v[0][0], v[0][1]), axis=0).flatten(), v[1]) for v in test])
    return train, test, val

def modelEval(model, test_df):
    test_x = test_df[0]
    test_y = test_df[1]

    test_x = np.array([a for a in np.array(test_x.values)])
    test_y = np.array([relatedness_conversion[int(a.max())] for a in np.array(test_y.values)])

    pred_y = model.predict(test_x)
    pred_y = (pred_y >= 0.5).astype(int)

    eval = ResultsDisplay(test_y, pred_y, ["unreated", "realted"])
    eval.metrics()
    eval.confusionMatrix()
    eval.rocCurve()



if __name__ == '__main__':
    model = "bert"
    training_size_limmit = 1000

    train, test, val = getDataLoaders(model=model, load=True, dataloader=False)
    train_df, test_df, val_df = dfList(train, test, val)

    #trained_model = ML_train(train_df, limmit=training_size_limmit)

    """with open("data/models/linReg_bert_1000.pkl", "wb") as file:
        pickle.dump(trained_model, file)"""
    with open("data/models/linReg_bert_1000.pkl", "rb") as file:
        trained_model = pickle.load(file)


    modelEval(trained_model, val_df)


