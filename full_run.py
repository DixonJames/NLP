from dl_classification import TrainingModel, LSTM
from bert_class import TrainingBertClass, BERTClassification
from utils import *
from datasets import getDataLoaders, getCombinedDataLoaders

import numpy as np
import pandas as pd
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

Stance_conversion = {"unrelated": 0, "agree": 1, "discuss": 3, "disagree": 2}
relatedness_conversion = {0: 0, 1: 1, 3: 1, 2: 1}


class FullClassification:
    def __init__(self, relatednessModel, discussionModel):
        self.relatednessModel = relatednessModel
        self.discussionModel = discussionModel

        self.tfidf_train, self.tfidf_test, self.tfidf_val = self.loadDataset("tfidf")
        self.bert_train, self.bert_test, self.bert_val = self.loadDataset("bert")

        self.labels = ["unrelated", "agree", "discuss", "disagree"]
        self.title = "LSTM & BERT Stack Classification"

    def batchSplitGen(self, batchIterator):
        for batch in batchIterator:
            labels = torch.tensor(batch[1]).float()
            heads, bodies = torch.split(batch[0], 1, dim=1)
            heads = heads.squeeze(dim=1)
            bodies = bodies.squeeze(dim=1)
            yield heads, bodies, labels

    def loadDataset(self, model):
        train, test, val = getCombinedDataLoaders(model=model, load=True, dataloader=True, shuffle_ds=False, balance=False)
        return train, test, val

    def evaluate(self):
        self.relatednessModel.eval()
        self.discussionModel.eval()

        all_realtedness_predictions = []
        all_discussion_predictions = []
        all_labels = []
        with torch.no_grad():
            # validation loop
            bert_val_it = self.batchSplitGen(self.bert_val)
            tfidf_val_it = self.batchSplitGen(self.tfidf_val)
            for (bert_heads, bert_bodies, bert_labels), (tfidf_heads, tfidf_bodies, tfidf_labels) in zip(bert_val_it,
                                                                                                         tfidf_val_it):
                bert_labels = bert_labels.to(device)
                bert_heads = bert_heads.to(device)
                bert_bodies = bert_bodies.to(device)

                tfidf_heads = tfidf_heads.to(device)
                tfidf_bodies = tfidf_bodies.to(device)

                relatedness_output = self.relatednessModel(tfidf_heads, tfidf_bodies)
                discussion_output = self.discussionModel(bert_heads, bert_bodies)

                with torch.no_grad():
                    all_realtedness_predictions.extend(list(relatedness_output.to("cpu").numpy()))
                    all_discussion_predictions.extend(list(discussion_output.to("cpu").numpy()))
                    all_labels.extend(list(bert_labels.to("cpu").numpy()))

        # get validation set accuracey

        all_relatedness_predictions_cateogries = [((p) >= 0.5).astype(int) for p in all_realtedness_predictions]
        all_discussion_predictions_cateogries = [list(p).index(max(list(p))) for p in all_discussion_predictions]
        final_predictions = []

        for r, d in zip(all_relatedness_predictions_cateogries, all_discussion_predictions_cateogries):
            if r == 0:
                final_predictions.append(r)
            else:
                final_predictions.append(d + 1)

        eval_res = ResultsDisplay(model=None, real_values=all_labels, test_df=None, labels=self.labels,
                                  title=self.title)
        eval_res.predicted_probs = final_predictions
        eval_res.predicted_categories = final_predictions

        eval_res.metrics()
        eval_res.confusionMatrix()
        # eval_res.rocCurve()

def runfull():
    bert_discussion = TrainingBertClass(model=BERTClassification, epochs=1, embedding_scheme="bert",
                                        save_folder=f"data/dl_classification",
                                        embeddings_len=None, title="Bert DL Classification",
                                        labels=["Agree", "Disagree", "Discuss"])
    bert_discussion.load_checkpoint()

    lstm_relatdedness = TrainingModel(model=LSTM, epochs=20, embedding_scheme="tfidf", save_folder=f"data/lstm/tfidf",
                                      embeddings_len=6622, title="tfidf", labels=["agree", "disagree", "discuss"])
    lstm_relatdedness.load_checkpoint()

    fc = FullClassification(relatednessModel=lstm_relatdedness.model, discussionModel=bert_discussion.model)
    fc.evaluate()


if __name__ == '__main__':
    bert_discussion = TrainingBertClass(model=BERTClassification, epochs=1, embedding_scheme="bert",
                                        save_folder=f"data/dl_classification",
                                        embeddings_len=None, title="Bert DL Classification",
                                        labels=["Agree", "Disagree", "Discuss"])
    bert_discussion.load_checkpoint()

    lstm_relatdedness = TrainingModel(model=LSTM, epochs=20, embedding_scheme="tfidf", save_folder=f"data/lstm/tfidf",
                                      embeddings_len=6622, title="tfidf", labels=["agree", "disagree", "discuss"])
    lstm_relatdedness.load_checkpoint()

    fc = FullClassification(relatednessModel=lstm_relatdedness.model, discussionModel=bert_discussion.model)
    fc.evaluate()
