# std libs
from pathlib import Path

# third party libs
import torch
import numpy as np
import matplotlib.pyplot as plt

# user defined libs
from .params import *


"""
load_data: loading data from dataset
  path: indicates the location of the file
  mode: selects which subject in the dataset, or all if not in [0, 9]
"""
def load_data(path, subject=-1):
  X_train_val = np.load(path / "X_train_valid.npy")
  y_train_val = np.load(path / "y_train_valid.npy")
  person_train_val = np.load(path / "person_train_valid.npy").reshape(-1)

  X_test = np.load(path / "X_test.npy")
  y_test = np.load(path / "y_test.npy")
  person_test = np.load(path / "person_test.npy").reshape(-1)

  y_train_val -= START_IDX
  y_test -= START_IDX

  if subject >= 0 and subject < 9:
    return X_train_val[person_train_val == subject], \
           y_train_val[person_train_val == subject], \
           X_test[person_test == subject], \
           y_test[person_test == subject], \
           person_train_val, \
           person_test

  return X_train_val, y_train_val, \
         X_test, y_test, person_train_val, person_test


def accuracy(y_pred, y_test):
  return torch.sum(torch.eq(y_pred, y_test)) / float(y_pred.size(0))


def plot(loss_history, acc_history):
  train_loss_history, val_loss_history = loss_history
  train_acc_history, val_acc_history = acc_history

  plt.subplot(2, 1, 1)
  plt.plot(train_loss_history, '-o')
  plt.plot(val_loss_history, '-o')
  plt.legend(['train', 'val'])
  plt.xlabel('epoch')
  plt.ylabel('loss')

  plt.subplot(2, 1, 2)
  plt.plot(train_acc_history, '-o')
  plt.plot(val_acc_history, '-o')
  plt.legend(['train', 'val'])
  plt.xlabel('epoch')
  plt.ylabel('accuracy')

  try:
    plt.draw()
    plt.pause(1e-3)
    input("Press enter to continue...")
    plt.close()
  except:
    pass


def print_train_batch(epoch, loss, batch_idx, X_batch, train_loader):
  loss /= len(X_batch)
  i = batch_idx * len(X_batch)
  total = len(train_loader.dataset)
  percentage = 100. * batch_idx / len(train_loader)
  print(f"{'train epoch':<12}{epoch:>3}: [{i:>4}/{total:<4} ({percentage:>2.0f}%)] loss={loss:.6f}")

def print_train_info(epoch, train_loss, train_acc=None):
  print(f"\n{'':<6}epoch {epoch:>3}: {'training':>17} {'loss':>4}={train_loss:<.6f}")
  if train_acc:
    print(f"{'':<6}epoch {epoch:>3}: {'training':>17} {'accu':>4}={train_acc:<.6f}")

def print_val_info(val_loss, val_acc=None):
  print(f"{'':<16} {'validation':>17} {'loss':>4}={val_loss:.6f}")
  if val_acc:
    print(f"{'':<16} {'validation':>17} {'accu':>4}={val_acc:.6f}\n")

