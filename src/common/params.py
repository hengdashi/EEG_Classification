START_IDX = 769

args = {
  # common arguments for all models
  "common": {
    # select which model to use (default: rnn)
    # selections are (cnn, crnn, vaecnn, vaernn)
    "model": "crnn",
    # random seed (default: 42)
    "seed": 42,
    # enables CUDA training (default: false)
    "cuda": True,
    # upsampling
    "scale": 1,
    # which subject to train (default: -1)
    "subject": -1,
    # k-fold cross validation (default: 5)
    "k_fold": 5,
    # enables plotting (default: false)
    "show_plot": False,
    # how many batches to wait before logging training status (default: 10)
    "log_interval": 10,
    # number of classes (moving part), always 4
    "n_classes": 4,
    # number of features (EEG electrodes), always 22
    "n_features": 22,
    # length of sequence (signals over time), always 1000
    "seq_len": 1000
  },
  # rnn parameters
  "crnn": {
    # number of epochs to train (default: 50)
    "epochs": 100,
    # out_channel (default: 40)
    "out_channel": 40,
    # kernel size (default: 20)
    "kernel": 20,
    # stride (default: 4)
    "stride": 4,
    # pooling size (default: 4)
    "pool": 4,
    # input batch size for training (default: 50)
    "batch_size": 100,
    # number of layers (default: 1)
    "n_layers": 3,
    # RNN layer type (default: rnn)
    # selections are (rnn, lstm, gru)
    "type": "lstm",
    # hidden dimension (default: 20)
    "hidden_dim": 64,
    # bidirectional (default: False)
    "bidirectional": True,
    # nonlinearity, only valid for rnn (default: tanh)
    # selections are (tanh, relu)
    "nonlinearity": "relu",
    # dropout rate (default: 0.5)
    "dropout": 0.5,
    # learning rate (default: 1e-4)
    "lr": 2e-3,
    # optimizer type (default: adam)
    # selections are (adam, rmsprop, lbfgs)
    "optimizer": "adam",
    # weight decay (default: 1e-2)
    "w_decay": 1e-2,
    # gradient clipping norm value (default: 0.25)
    "clip": 0.5
  },

  "vae": {
    # number of epochs to train (default: 50)
    "epochs": 50,
    # learning rate (default: 1e-5)
    "lr": 1e-5,
    # whether to save the state of vae or not (default: False)
    "save": True
  },

  "cnn": {
      # number of epochs to train (default: 10)
    "epochs": 50,
    # input batch size for training (default: 50)
    "batch_size": 50,
    # number of layers (default: [3, 4, 6, 3])
    "layers": [3, 4, 6, 3],
    # learning rate (default: 1e-3)
    "lr": 1e-4,
    # weight decay (default: 1e-2)
    "w_decay": 1e-2,
    # dropout rate (default: 0.5)
    "dropout": 0.5,
    # kernel size (default: 20)
    "kernel": 20,
    # pool size (default: 6)
    "pool": 6
  }
}
