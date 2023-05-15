from dl_classification import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import getRelatedDataLoaders


# Stance_conversion = {"unrelated": 0, "agree": 1, "discuss": 3, "disagree": 2}

class BERTClassification(nn.Module):
    """
    modified from [https://www.kaggle.com/code/joydeb28/text-classification-with-bert-pytorch]
    """

    def __init__(self, embeddings_length, hidden_dim_len=1024):
        super(BERTClassification, self).__init__()



        self.fc1 = nn.Linear(768 * 2, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 3)
        self.sigmoid_layer = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, head, body):
        x = torch.cat((head, body), dim=1).float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid_layer(x)
        x = self.softmax(x)
        return x



class TrainingBertClass(TrainingModel):
    """
    some methods modified from [https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0]
    """

    def __init__(self, model, epochs, embedding_scheme, save_folder="data/lstm", embeddings_len=768,
                 labels=["related", "unrelated"], title="LSTM"):

        super(TrainingBertClass, self).__init__(model=model, epochs=epochs, embedding_scheme=embedding_scheme,
                                                save_folder=save_folder, embeddings_len=embeddings_len,
                                                labels=labels, title=title)

        self.raw_train = None
        self.loadDataset()

        self.model = model(embeddings_length=embeddings_len, hidden_dim_len=1024).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.criterion = nn.CrossEntropyLoss(weight=self.getClassWeights()).to(device)

        self.evaluation_gap = 1000

    def getClassWeights(self):
        class_proportions = pd.Series(
            (np.array([list(t[1].numpy().flatten()) for t in self.test_iterator][:-1]).flatten())).value_counts(
            normalize=True)
        class_weights = 1.0 / class_proportions
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor([class_weights[1], class_weights[2], class_weights[3]]).float().to(device)
        return class_weights

    def loadDataset(self):
        train, test, val = getRelatedDataLoaders(model=self.embedding_schemem, load=True, dataloader=True)
        self.raw_train = train
        self.train_iterator_len = len(train)
        self.test_iterator_len = len(test)
        self.validate_iterator_len = len(val)

        self.train_iterator = train
        self.test_iterator = test
        self.validate_iterator = val

    def batchSplitGen(self, batchIterator):
        for batch in batchIterator:
            labels = torch.tensor([b.numpy().max() - 1 for b in batch[1]]).float()
            heads, bodies = torch.split(batch[0], 1, dim=1)
            heads = heads.squeeze(dim=1)
            bodies = bodies.squeeze(dim=1)
            yield heads, bodies, labels

    def evaluate(self, global_step, plots=False):
        self.model.eval()

        all_predictions = []
        all_labels = []
        with torch.no_grad():
            # validation loop
            val_it = self.batchSplitGen(self.validate_iterator)
            for heads, bodies, labels in val_it:
                labels = labels.long().to(device)
                heads = heads.to(device)
                bodies = bodies.to(device)

                output = self.model(heads, bodies)

                with torch.no_grad():
                    all_predictions.extend(list(output.to("cpu").numpy()))
                    all_labels.extend(list(labels.to("cpu").numpy()))

                loss = self.criterion(output, labels)
                self.valid_running_loss += loss.item()

        # evaluation
        average_train_loss = self.running_loss / self.evaluation_gap
        average_valid_loss = self.valid_running_loss / self.validate_iterator_len

        # get validation set accuracey
        correct_eval_predictions = [list(p).index(max(list(p))) for p in all_predictions]
        acc = sum(1 for p, l in zip(correct_eval_predictions, all_labels) if int(p) == int(l)) / len(
            correct_eval_predictions)
        self.test_acc.append(acc)

        if not plots:
            self.train_loss_list.append(average_train_loss)
            self.valid_loss_list.append(average_valid_loss)
            print(f"loss: train:{average_train_loss}, val:{average_valid_loss}")

            self.global_steps_list.append(global_step)

            # resetting running values
            self.running_loss = 0.0
            self.valid_running_loss = 0.0
            self.model.train()

            # print progress
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                  .format(self.epoch + 1, self.epoch_limmit, global_step, self.epoch_limmit * self.train_iterator_len,
                          average_train_loss, average_valid_loss))

            # checkpoint
            if self.best_valid_loss > average_valid_loss:
                self.best_valid_loss = average_valid_loss
                self.save_checkpoint(self.save_folder + '/model.pt')
                self.save_metrics(self.save_folder + '/metrics.pt')

        else:
            eval_res = ResultsDisplay(model=None, real_values=all_labels, test_df=None, labels=self.labels,
                                      title=self.title)
            eval_res.predicted_probs = all_predictions
            eval_res.predicted_categories = correct_eval_predictions

            # eval_res.metrics()
            eval_res.confusionMatrix()
            # eval_res.rocCurve()

    def train(self):
        """
        modified from [https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0]
        """
        train_start = time.time()
        # initialize running values
        self.running_loss = 0.0
        self.valid_running_loss = 0.0
        global_step = 0

        # training loop
        self.model.train()
        for epoch in range(self.epoch_limmit):
            epoch_step = 0
            train_it = self.batchSplitGen(self.train_iterator)

            all_predictions = []
            all_labels = []
            for heads, bodies, labels in train_it:

                labels = labels.long().to(device)
                heads = heads.to(device)
                bodies = bodies.to(device)

                output = self.model(heads, bodies)

                with torch.no_grad():
                    all_predictions.extend(list(output.to("cpu").numpy()))
                    all_labels.extend(list(labels.to("cpu").numpy()))

                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update running values
                self.running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % self.step_save_gap == 0:
                    correct_predictions = [list(p).index(max(list(p))) for p in all_predictions]
                    acc = sum(1 for p, l in zip(correct_predictions, all_labels) if int(p) == int(l)) / len(
                        correct_predictions)

                    self.train_acc.append(acc)
                    self.evaluate(global_step)

                print(f"Epoch: {epoch}, {(epoch_step / self.train_iterator_len) * 100}%")
                epoch_step += 1

            time_done = time.time() - train_start
            done_proportion = epoch + 1 / self.epoch_limmit
            print(f"time_remaining: {(1 / done_proportion) * time_done}")

        self.save_metrics(self.save_folder + '/metrics.pt')
        print('Finished Training!')


if __name__ == '__main__':
    bert_classifier = TrainingBertClass(model=BERTClassification, epochs=1, embedding_scheme="bert",
                                        save_folder=f"data/dl_classification",
                                        embeddings_len=None, title="Bert DL Classification",
                                        labels=["Agree", "Disagree", "Discuss"])
    bert_classifier.train()
    bert_classifier.evaluate(0, plots=True)
