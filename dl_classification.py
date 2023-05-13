"""
question 2)a)ii)

this code is modified from [https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0]
"""

import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import time
from datasets import getDataLoaders
from utils import ResultsDisplay

relatedness_conversion = {0: 0, 1: 1, 3: 1, 2: 1}

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, embeddings_length, hidden_dim_len=1024):
        super(LSTM, self).__init__()
        self.hidden_dimension_length = hidden_dim_len
        self.embeddings_length = embeddings_length

        self.fc1 = nn.Linear(2 * embeddings_length, self.hidden_dimension_length)
        self.fc2 = nn.Linear(self.hidden_dimension_length*2, 1)

        self.lstm = nn.LSTM(input_size=self.hidden_dimension_length,
                            hidden_size=self.hidden_dimension_length,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, head, body):
        embedding = torch.cat((head, body), dim=1)
        x = self.fc1(embedding)


        x, _ = self.lstm(x)


        out_forward = x[range(len(x)), : self.hidden_dimension_length]
        out_reverse = x[:,  self.hidden_dimension_length:]

        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc2(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


class TrainingLSTM:
    """
    some methods modified from [https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0]
    """

    def __init__(self, model, epochs, embedding_scheme, save_folder="data/lstm", embeddings_len=768):

        self.embedding_schemem = embedding_scheme
        self.epoch_limmit = epochs
        self.epoch = 0

        self.save_folder = save_folder

        self.train_iterator = None
        self.test_iterator = None
        self.validate_iterator = None

        self.train_iterator_len = None
        self.test_iterator_len = None
        self.validate_iterator_len = None

        self.train_loss_list = []
        self.valid_loss_list = []
        self.global_steps_list = []

        self.running_loss = 0.0
        self.valid_running_loss = 0.0
        self.best_valid_loss = float("Inf")

        self.loadDataset()

        self.model = model(embeddings_length=embeddings_len, hidden_dim_len=1024).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

        self.evaluation_gap = 1000

    def batchSplitGen(self, batchIterator):
        for batch in batchIterator:
            labels = torch.tensor([relatedness_conversion[(b.numpy().max())] for b in batch[1]]).float()
            heads, bodies = torch.split(batch[0], 1, dim=1)
            heads = heads.squeeze(dim=1)
            bodies = bodies.squeeze(dim=1)
            yield heads, bodies, labels

    def loadDataset(self):
        train, test, val = getDataLoaders(model=self.embedding_schemem, load=True, dataloader=True)
        self.train_iterator_len = len(train)
        self.test_iterator_len = len(test)
        self.validate_iterator_len = len(val)
        train, test, val = self.batchSplitGen(train), self.batchSplitGen(test), self.batchSplitGen(val)
        self.train_iterator = train
        self.test_iterator = test
        self.validate_iterator = val

    def save_checkpoint(self, save_path):
        state_dict = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'valid_loss': self.best_valid_loss}

        torch.save(state_dict, save_path)

    def load_checkpoint(self):

        state_dict = torch.load(self.save_folder + '/metrics.pt', map_location=device)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def save_metrics(self, save_path):
        state_dict = {'train_loss_list': self.train_loss_list,
                      'valid_loss_list': self.valid_loss_list,
                      'global_steps_list': self.global_steps_list}
        torch.save(state_dict, save_path)

    def load_metrics(self, load_path):
        state_dict = torch.load(load_path, map_location=device)
        return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

    def evaluate(self, global_step):
        self.model.eval()

        with torch.no_grad():
            # validation loop
            for heads, bodies, labels in self.validate_iterator:
                labels = labels.to(device)
                heads = heads.to(device)
                bodies = bodies.to(device)

                output = self.model(heads, bodies)

                loss = self.criterion(output, labels)
                self.valid_running_loss += loss.item()

        # evaluation
        average_train_loss = self.running_loss / self.evaluation_gap
        average_valid_loss = self.valid_running_loss / self.validate_iterator_len

        self.train_loss_list.append(average_train_loss)
        self.valid_loss_list.append(average_valid_loss)

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
            for heads, bodies, labels in self.train_iterator:

                labels = labels.to(device)
                heads = heads.to(device)
                bodies = bodies.to(device)

                output = self.model(heads, bodies)

                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update running values
                self.running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % self.evaluation_gap == 0:
                    self.evaluate(global_step)

                print(f"Epoch: {epoch}, {(epoch_step / self.train_iterator_len) * 100}%")
                epoch_step += 1

            time_done = time.time() - train_start
            done_proportion = epoch+1/self.epoch_limmit
            print(f"time_remaining: {(1/done_proportion) * time_done}")


        self.save_metrics(self.save_folder + '/metrics.pt')
        print('Finished Training!')


if __name__ == '__main__':
    model_type = "bert"
    lstm = TrainingLSTM(model=LSTM, epochs=50, embedding_scheme=model_type, save_folder="data/lstm")
    lstm.train()
