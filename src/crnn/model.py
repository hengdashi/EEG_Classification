import math

# third party libs
import torch.nn as nn

class CRNN(nn.Module):
  def __init__(self, args):
    super(CRNN, self).__init__()
    self.n_features = args['common']['n_features']
    self.seq_len = args['common']['seq_len']
    self.hidden_dim = args['crnn']['hidden_dim']
    self.layer_dim = args['crnn']['n_layers']
    self.output_dim = args['common']['n_classes']
    self.out_channel = args['crnn']['out_channel']
    self.kernel = args['crnn']['kernel']
    self.stride = args['crnn']['stride']
    self.pool = args['crnn']['pool']
    self.transform_dim = math.ceil(((math.ceil((self.seq_len - self.kernel - 2) / self.stride) + 1) - self.pool - 2) / self.stride) + 1

    self.num_directions = 2 if args['crnn']['bidirectional'] else 1

    if self.layer_dim == 1:
      self.dropout = 0
    else:
      self.dropout = args['crnn']['dropout']

    # conv1d
    self.conv = nn.Conv1d(self.n_features, self.out_channel, self.kernel, stride=self.stride)
    # relu
    self.relu = nn.ReLU()
    # dropout 1
    self.do1 = nn.Dropout(self.dropout)
    # batch norm 1
    self.bn1 = nn.BatchNorm1d(self.out_channel)
    # max pooling 1
    self.maxpool = nn.MaxPool1d(self.pool, self.stride)

    if args['crnn']['type'] == 'rnn':
      self.rnn = nn.RNN(self.out_channel, self.hidden_dim, self.layer_dim, nonlinearity=args['crnn']['nonlinearity'], batch_first=True, dropout=self.dropout, bidirectional=args['crnn']['bidirectional'])
    elif args['crnn']['type'] == 'lstm':
      self.rnn = nn.LSTM(self.out_channel, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout, bidirectional=args['crnn']['bidirectional'])
    elif args['crnn']['type'] == 'gru':
      self.rnn = nn.GRU(self.out_channel, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout, bidirectional=args['crnn']['bidirectional'])

    self.batchnorm = nn.BatchNorm1d(self.num_directions * self.hidden_dim)

    if self.layer_dim == 1:
      self.dropout_layer = nn.Dropout(self.dropout)

    self.fc1 = nn.Linear(self.num_directions * self.hidden_dim * self.transform_dim, self.hidden_dim)

    self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    # init weights
    self.reset_parameters()


  def forward(self, input_data):
    batch_size = input_data.size(0)
    # conv (batch_size, channel: 22, -1)
    output_data = self.conv(input_data)
    # relu (batch_size, channel: 40, -1)
    output_data = self.relu(output_data)
    # dropout 1 (batch_size, channel: 40, -1)
    output_data = self.do1(output_data)
    # batch norm 1 (batch_size, channel: 40, -1)
    output_data = self.bn1(output_data)
    # max pooling 1 (batch_size, channel: 40, -1)
    output_data = self.maxpool(output_data)
    # passing in the input and hidden state into the model and obtaining outputs
    output_data = output_data.permute(0, 2, 1)
    # lstm (batch_size, seq_len, n_features)
    output_data, hidden_data = self.rnn(output_data)

    # reshaping data to fit into batchnorm
    output_data = output_data.permute(0, 2, 1)
    # batch norm (batch_size, channel, -1)
    output_data = self.batchnorm(output_data)

    # reshape to fit into fc
    output_data = output_data.reshape(batch_size, -1)

    # go through dropout layer 
    if self.layer_dim == 1:
      output_data = self.dropout_layer(output_data)

    # go through fully connected layer
    output_data = self.fc1(output_data)  # batch_size, num_classes
    output_data = self.fc2(output_data)  # batch_size, num_classes

    return output_data, hidden_data

  """
  TODO: weight initialization
  """
  def reset_parameters(self):
    for m in self.modules():
      if type(m) == nn.RNN:
        for name, param in m.named_parameters():
          if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
          elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
          elif 'bias' in name:
            param.data.fill_(0)
