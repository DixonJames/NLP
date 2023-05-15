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

        self.train, self.test, self.val = self.loadDataset("tfidf")


        self.labels = ["unrelated", "agree", "discuss", "disagree"]
        self.title = "LSTM & BERT Stack Classification"

    def batchSplitGen(self, batchIterator):
        for batch in batchIterator:
            labels = torch.tensor(batch[2]).float()
            bert = batch[0]
            tfidf = batch[1]

            b_heads, b_bodies = torch.split(bert, 1, dim=1)
            b_heads = b_heads.squeeze(dim=1)
            b_bodies = b_bodies.squeeze(dim=1)

            t_heads, t_bodies = torch.split(tfidf, 1, dim=1)
            t_heads = t_heads.squeeze(dim=1)
            t_bodies = t_bodies.squeeze(dim=1)

            yield (b_heads, b_bodies), (t_heads, t_bodies), labels

    def loadDataset(self, model):
        train, test, val = getCombinedDataLoaders(model=model, load=True, dataloader=True, shuffle_ds=True, balance=True)
        return train, test, val

    def evaluate(self):
        self.relatednessModel.eval()
        self.discussionModel.eval()

        all_realtedness_predictions = []
        all_discussion_predictions = []
        all_labels = []
        with torch.no_grad():
            # validation loop
            val_it = self.batchSplitGen(self.val)

            for (bert_heads, bert_bodies), (tfidf_heads, tfidf_bodies), labels in val_it:
                labels = labels.to(device)

                bert_heads = bert_heads.to(device)
                bert_bodies = bert_bodies.to(device)

                tfidf_heads = tfidf_heads.to(device)
                tfidf_bodies = tfidf_bodies.to(device)

                relatedness_output = self.relatednessModel(tfidf_heads, tfidf_bodies)
                discussion_output = self.discussionModel(bert_heads, bert_bodies)

                with torch.no_grad():
                    all_realtedness_predictions.extend(list(relatedness_output.to("cpu").numpy()))
                    all_discussion_predictions.extend(list(discussion_output.to("cpu").numpy()))
                    all_labels.extend(list(labels.to("cpu").numpy()))

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
    runfull()
