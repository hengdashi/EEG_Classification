#!/usr/bin/env python3

# standard libs
import sys
import time
from datetime import timedelta
from pathlib import Path

# third party libs
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# self-defined libs
from common.params import *
from common.utils import load_data, accuracy
from crnn.classifier import CRNNClassifier
from cnn.classifier import CNNClassifier
from vae.embedding import VAEEmbedding

"""
TODO
1. augment data to enlarge dataset
5. second order method  (no enough gpu memory)

VAE+RNN two stage training may be the default option. Similar architecture was also found in figure 5 of this paper for example. 
#http://e-journal.uum.edu.my/index.php/jict/article/view/8253/1209
"""


if __name__ == "__main__":
  # preset settings
  if len(sys.argv) == 3 and sys.argv[1] == '--model':
    args['common']['model'] = sys.argv[2]

  model = args['common']['model']
  data_path = Path('../data/project')

  args['common']['cuda'] = args['common']['cuda'] and torch.cuda.is_available()
  args['common']['device'] = torch.device("cuda" if args['common']['cuda'] else "cpu")

  print("Using GPU" if args['common']['cuda'] else "Using CPU")


  # start loading data
  X_train_val, y_train_val, X_test, y_test, person_train_val, person_test = load_data(data_path)

  # standarize dataset
  scaler = StandardScaler()

  X_train_val = scaler.fit_transform(X_train_val.reshape(-1, X_train_val.shape[-1])).reshape(X_train_val.shape)
  # note that only use transform here because training dataset is larger
  X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

  # upsample data
  if args['common']['scale'] != 1:
    scale = args['common']['scale']
    X_train_val = Tensor(X_train_val)
    X_test = Tensor(X_test)
    X_train_val = F.interpolate(X_train_val, scale_factor=scale)
    X_test = F.interpolate(X_test, scale_factor=scale)
    args['common']['seq_len'] = X_train_val.size(2)
    X_train_val = X_train_val.numpy()
    X_test = X_test.numpy()


  # run VAE first
  if 'vae' in model:
    print("####################################################")
    if model == 'vaecnn':
      print(f"Running VAE+CNN: VAE STAGE")
    elif model == 'vaecrnn':
      print(f"Running VAE+RNN: VAE STAGE")
    elif model == 'vae':
      print(f"Running VAE")
    else:
      print("UNKNOWN MODEL: EXITING...")
      exit(1)

    X_train_vae, X_val_vae, y_person_train_vae, y_person_val_vae = train_test_split(X_train_val, person_train_val, test_size=0.33, random_state=args['common']['seed'])

    # send to device
    X_train_vae = Tensor(X_train_vae).to(args['common']['device'])
    y_person_train_vae = Tensor(y_person_train_vae).to(args['common']['device'])

    X_val_vae = Tensor(X_val_vae).to(args['common']['device'])
    y_person_val_vae = Tensor(y_person_val_vae).to(args['common']['device'])

    train_loader = DataLoader(TensorDataset(X_train_vae, y_person_train_vae), batch_size=50, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_vae, y_person_val_vae), batch_size=50, shuffle=True)

    if args['common']['show_plot']:
      imgplot1 = plt.imshow(X_train_val[1])

    # train and generate embedding
    emb = VAEEmbedding(args, train_loader, val_loader)
    emb.fit()
    emb.save_state()
    X_train_val = emb.predict(X_train_val)
    # free up memory
    emb.detach()


  # run CNN or RNN if one of them is specified
  if 'cnn' in model or 'crnn' in model:
    print("####################################################")
    if model == 'vaecnn':
      print(f"Running VAE+CNN: CNN STAGE")
    elif model == 'vaecrnn':
      print(f"Running VAE+CRNN: RNN STAGE")
    elif model == 'cnn':
      print(f"Running CNN")
    elif model == 'crnn':
      print(f"Running CRNN")
    else:
      print("UNKNOWN MODEL: EXITING...")
      exit(1)

    kwargs = {"args": args, "X_train_val": X_train_val, "y_train_val": y_train_val}

    if 'cnn' in args['common']['model']:
      clf = CNNClassifier(**kwargs)
    elif 'crnn' in args['common']['model']:
      clf = CRNNClassifier(**kwargs)

    start = time.time()
    clf.fit()
    end = time.time()
    elapsed = end - start
    print(f"Total Training Time: {timedelta(seconds=elapsed)}")

    X_test = torch.Tensor(X_test).to(args['common']['device'])
    y_test = torch.Tensor(y_test).long().to(args['common']['device'])

    y_pred = clf.predict(X_test)
    print(f"Test Accuracy: {accuracy(y_pred, y_test):.4f}")
