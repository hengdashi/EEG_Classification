# third party libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold

# user-defined libs
from common.utils import *
from .model import BasicBlock, ResNet, DeepConvNet


class CNNClassifier():
  def __init__(self, args, X_train_val, y_train_val):
    self.args = args
    self.X_train_val = X_train_val
    self.y_train_val = y_train_val

    # self.cnn = ResNet(BasicBlock, args['cnn']['layers']).to(args['common']['device'])
    self.cnn = DeepConvNet(args).to(args['common']['device'])

    self.skf = StratifiedKFold(n_splits=args['common']['k_fold'], shuffle=True, random_state=args['common']['seed'])

    self.softmax = nn.LogSoftmax(dim=1).to(self.args['common']['device'])
    self.criterion = nn.CrossEntropyLoss().to(self.args['common']['device'])

    self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.args['cnn']['lr'])

  def train(self, epoch, train_loader):
    self.cnn.train()

    train_loss = 0
    train_acc = 0

    X_train = torch.Tensor().to(self.args['common']['device'])
    y_train = torch.Tensor().long().to(self.args['common']['device'])

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
      # save dataset
      X_train = torch.cat((X_train, X_batch), 0)
      y_train = torch.cat((y_train, y_batch), 0)

      self.optimizer.zero_grad()
      y_scores = self.cnn(X_batch)
      loss = self.criterion(y_scores, y_batch)
      loss.backward()

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
    self.cnn.eval()

    val_loss = 0
    val_acc = 0

    X_val = torch.Tensor().to(self.args['common']['device'])
    y_val = torch.Tensor().long().to(self.args['common']['device'])

    with torch.no_grad():
      for i, (X_batch, y_batch) in enumerate(val_loader):
        # save dataset
        X_val = torch.cat((X_val, X_batch), 0)
        y_val = torch.cat((y_val, y_batch), 0)
        # feed into cnn
        y_scores = self.cnn(X_batch)
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
    epochs_fold = self.args['cnn']['epochs'] // self.skf.get_n_splits(self.X_train_val, self.y_train_val)
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
      train_loader = DataLoader(train_dataset, batch_size=self.args['cnn']['batch_size'], shuffle=True)
      val_loader   = DataLoader(val_dataset, batch_size=self.args['cnn']['batch_size'], shuffle=True)

      for epoch in range(1, epochs_fold+1):
        train_loss, train_acc = self.train(epoch, train_loader)
        val_loss, val_acc = self.validate(val_loader)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    if self.args['common']['show_plot']:
      plot((train_loss_history, val_loss_history), (train_acc_history, val_acc_history))


  def predict(self, X_test):
    self.cnn.eval()

    with torch.no_grad():
      # reshape data to fit into the model
      y_scores = self.cnn(X_test)
      _, y_pred = torch.max(self.softmax(y_scores), dim=1)

    return y_pred
