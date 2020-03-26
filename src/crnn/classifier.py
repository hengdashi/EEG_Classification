# std libs
import time

# third party libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold

# user defined libs
from .model import CRNN
from common.utils import *


class CRNNClassifier():
  def __init__(self, args, X_train_val, y_train_val):
    self.args = args

    self.X_train_val = X_train_val
    self.y_train_val = y_train_val
    self.n_features = self.args['common']['n_features']
    self.seq_len = self.args['common']['seq_len']

    # k-fold splitter
    self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args['common']['seed'])
    # rnn
    self.model = CRNN(args).to(self.args['common']['device'])
    # define loss and optimizer
    self.softmax = nn.LogSoftmax(dim=1).to(args['common']['device'])
    # loss function
    self.criterion = nn.CrossEntropyLoss().to(args['common']['device'])
    # optimizer
    if args['crnn']['optimizer'] == 'adam':
      self.optimizer = optim.Adam(self.model.parameters(), lr=args['crnn']['lr'], weight_decay=args['crnn']['w_decay'])
    elif args['crnn']['optimizer'] == 'rmsprop':
      self.optimizer = optim.RMSprop(self.model.parameters(), lr=args['crnn']['lr'], weight_decay=args['crnn']['w_decay'])
    elif args['crnn']['optimizer'] == 'lbfgs':
      self.optimizer = optim.LBFGS(self.model.parameters(), lr=args['crnn']['lr'])
    # lr scheduler
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

  def train(self, epoch, train_loader):
    # set to training mode
    self.model.train()

    # closure function for lbfgs
    def closure():
      nonlocal train_loss
      nonlocal lossitem
      nonlocal y_scores
      self.optimizer.zero_grad()
      y_scores, _ = self.model(X_batch)
      loss = self.criterion(y_scores, y_batch)
      loss.backward()
      lossitem = loss.item()
      train_loss += loss.item()
      return loss

    train_loss = 0
    train_acc = 0

    X_train = torch.Tensor().to(self.args['common']['device'])
    y_train = torch.Tensor().long().to(self.args['common']['device'])

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
      lossitem = 0

      # save dataset
      X_train = torch.cat((X_train, X_batch), 0)
      y_train = torch.cat((y_train, y_batch), 0)

      if self.args['crnn']['optimizer'] == 'lbfgs':
        self.optimizer.step(closure)
      else:
        self.optimizer.zero_grad()
        # pass in model
        y_scores, hidden = self.model(X_batch)
        # compute loss function
        loss = self.criterion(y_scores, y_batch)
        # backprop
        loss.backward()
        # keep gradient in range
        # clip_grad_norm_(self.model.parameters(), self.args['crnn']['clip'])
        train_loss += loss.item()
        self.optimizer.step()

      # print progress
      if not batch_idx % self.args['common']['log_interval']:
        print_train_batch(epoch, loss.item(), batch_idx, X_batch, train_loader)

    train_loss /= len(train_loader.dataset)
    train_acc = accuracy(self.predict(X_train), y_train)
    print_train_info(epoch, train_loss, train_acc)

    return train_loss, train_acc


  def validate(self, val_loader):
    # set to validation mode
    self.model.eval()

    val_loss = 0
    val_acc = 0

    X_val = torch.Tensor().to(self.args['common']['device'])
    y_val = torch.Tensor().long().to(self.args['common']['device'])

    with torch.no_grad():
      for i, (X_batch, y_batch) in enumerate(val_loader):
        # save dataset
        X_val = torch.cat((X_val, X_batch), 0)
        y_val = torch.cat((y_val, y_batch), 0)
        # feed into crnn
        y_scores, _ = self.model(X_batch)
        # accumulate loss
        val_loss += self.criterion(y_scores, y_batch).item()

    val_loss /= len(val_loader.dataset)
    val_acc = accuracy(self.predict(X_val), y_val)
    print_val_info(val_loss, val_acc)

    return val_loss, val_acc


  def fit(self):
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    # k-fold stratified cross validation
    epochs_fold = self.args['crnn']['epochs'] // self.skf.get_n_splits(self.X_train_val, self.y_train_val)
    for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(self.X_train_val, self.y_train_val)):
      print(f"train fold {fold_idx+1:>4}")


      X_train_fold, X_val_fold = self.X_train_val[train_idx], self.X_train_val[val_idx]
      y_train_fold, y_val_fold = self.y_train_val[train_idx], self.y_train_val[val_idx]

      X_train_fold = torch.Tensor(X_train_fold).to(self.args['common']['device'])
      y_train_fold = torch.Tensor(y_train_fold).long().to(self.args['common']['device'])
      X_val_fold   = torch.Tensor(X_val_fold).to(self.args['common']['device'])
      y_val_fold   = torch.Tensor(y_val_fold).long().to(self.args['common']['device'])

      train_dataset = TensorDataset(X_train_fold, y_train_fold)
      val_dataset   = TensorDataset(X_val_fold, y_val_fold)
      # lbfgs only works for full-batch training
      if self.args['crnn']['optimizer'] == 'lbfgs':
        self.args['crnn']['batch_size'] = len(train_dataset)
      train_loader = DataLoader(train_dataset, batch_size=self.args['crnn']['batch_size'], shuffle=True)
      val_loader   = DataLoader(val_dataset, batch_size=self.args['crnn']['batch_size'], shuffle=True)

      for epoch in range(1, epochs_fold+1):
        train_loss, train_acc = self.train(epochs_fold * fold_idx + epoch, train_loader)
        val_loss, val_acc = self.validate(val_loader)
        self.scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    if self.args['common']['show_plot']:
      plot((train_loss_history, val_loss_history), (train_acc_history, val_acc_history))


  def predict(self, X_test):
    # set to validation mode
    self.model.eval()

    with torch.no_grad():
      y_scores, _ = self.model(X_test)
      _, y_pred = torch.max(self.softmax(y_scores), dim=1)

    return y_pred
