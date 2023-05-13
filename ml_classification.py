import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
from datasets import getDataLoaders
import numpy as np
from utils import ResultsDisplay

relatedness_conversion = {0: 0, 1: 1, 3: 1, 2: 1}


# pca = PCA(n_components=0.95)

def ML_train(train, limmit=0):
    train_x = train[0]
    train_y = train[1]

    train_x = np.array([a for a in np.array(train_x.values)])
    train_y = np.array([relatedness_conversion[int(a.max())] for a in np.array(train_y.values)])

    if limmit != 0:
        train_x = train_x[:limmit]
        train_y = train_y[:limmit]

    linear_regression = LogisticRegression(max_iter=10000)

    linear_regression.fit(train_x, train_y)

    return linear_regression


def dfList(train, test, val):
    train = pd.DataFrame([(np.concatenate((v[0][0], v[0][1]), axis=0).flatten(), v[1]) for v in train])
    val = pd.DataFrame([(np.concatenate((v[0][0], v[0][1]), axis=0).flatten(), v[1]) for v in val])
    test = pd.DataFrame([(np.concatenate((v[0][0], v[0][1]), axis=0).flatten(), v[1]) for v in test])
    return train, test, val


def predict(model, test_df):
    test_x = test_df[0]
    # test_y = test_df[1]

    test_x = np.array([a for a in np.array(test_x.values)])
    # test_y = np.array([relatedness_conversion[int(a.max())] for a in np.array(test_y.values)])

    pred_y = model.predict(test_x)
    pred_y = (pred_y >= 0.5).astype(int)
    return pred_y


def modelEval(model, test_df, title):
    true_labels = np.array([relatedness_conversion[l] for l in test_df[1].values.tolist()])
    # pred_y = predict(model, test_df)

    eval = ResultsDisplay(model=model, real_values=true_labels, test_df=test_df, labels=["Related", "Unrelated"],
                          title=title)
    eval.metrics()
    eval.confusionMatrix()
    eval.rocCurve()


def trainAll(training_size_limmit=1000):
    for model in ["bert", "tfidf"]:
        train, test, val = getDataLoaders(model=model, load=True, dataloader=False)
        train_df, test_df, val_df = dfList(train, test, val)

        trained_model = ML_train(train_df, limmit=training_size_limmit)

        with open(f"data/{model}/logReg_{training_size_limmit}.pkl", "wb") as file:
            pickle.dump(trained_model, file)



def evaluateAll(training_size_limmit=1000):
    for model in ["bert", "tfidf"]:
        train, test, val = getDataLoaders(model=model, load=True, dataloader=False)
        train_df, test_df, val_df = dfList(train, test, val)

        with open(f"data/{model}/logReg_{training_size_limmit}.pkl", "rb") as file:
            trained_model = pickle.load(file)

        # predicted = predict(trained_model, val_df)

        modelEval(trained_model, val_df, title=f"Logistic Regression - {model}")


if __name__ == '__main__':
    trainAll(training_size_limmit=20000)
    #evaluateAll(training_size_limmit=1000)
