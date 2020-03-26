# third party libs
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
  def __init__(self, args):
    super(VAE, self).__init__()

    self.args = args
    self.fc1 = nn.Linear(22000, 2000)
    self.fc21 = nn.Linear(2000, 2000)
    self.fc22 = nn.Linear(2000, 2000)
    self.fc3 = nn.Linear(2000, 5000)
    self.fc4 = nn.Linear(5000, 22000)

  def encode(self, x):
    h1 = F.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return self.fc4(h3)

  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, 22000))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar
