# std libs
from pathlib import Path

# third party libs
import torch
import torch.optim as optim
from torchvision.utils import save_image

# user-defined libs
from .model import VAE
from .optimizer import loss_function

from common.params import *
from common.utils import *


class VAEEmbedding():
  def __init__(self, args, train_loader, val_loader):
    self.args = args
    self.train_loader = train_loader
    self.val_loader = val_loader

    self.vae = VAE(args).to(self.args['common']['device'])
    self.optimizer = optim.Adam(self.vae.parameters(), lr=self.args['vae']['lr'])


  def train(self, epoch):
    self.vae.train()

    train_loss = 0
    for batch_idx, (X_batch, _) in enumerate(self.train_loader):
      self.optimizer.zero_grad()
      recon_batch, mu, logvar = self.vae(X_batch)
      loss = loss_function(recon_batch, X_batch, mu, logvar, self.args['common']['scale'])
      loss.backward()
      lossitem = loss.item()
      train_loss += loss.item()
      self.optimizer.step()

      if not batch_idx % args['common']['log_interval']:
        print_train_batch(epoch, loss.item(), batch_idx, X_batch, self.train_loader)

    train_loss /= len(self.train_loader.dataset)
    print_train_info(epoch, train_loss)


  def validate(self, epoch):
    self.vae.eval()

    val_loss = 0
    image_path = Path('./results/vae')

    with torch.no_grad():
      for i, (X_batch, _) in enumerate(self.val_loader):
        recon_batch, mu, logvar = self.vae(X_batch)
        val_loss += loss_function(recon_batch, X_batch, mu, logvar, self.args['common']['scale']).item()
        if i == 0:
          n = min(X_batch.size(0), 8)
          comparison = torch.cat([X_batch.view(-1, 1, 22, self.args['common']['scale'] * 1000)[:n],
                                recon_batch.view(50, 1, 22, self.args['common']['scale'] * 1000)[:n]]) #batch_size=50...

          if args['vae']['save']:
            if not image_path.exists():
                image_path.mkdir(parents=True)
            save_image(comparison.cpu(),
                      f'{str(image_path)}/reconstruction_{epoch}.png', nrow=n)

    val_loss /= len(self.val_loader.dataset)
    print_val_info(val_loss)


  def fit(self):
    for epoch in range(self.args['vae']['epochs']):
      self.train(epoch)
      self.validate(epoch)


  def predict(self, X_test):
    if not torch.is_tensor(X_test):
      X_test = torch.Tensor(X_test).to(self.args['common']['device'])
    # set to validation mode
    self.vae.eval()

    with torch.no_grad():
      # reshape data to fit into the model
      X_recon, _, _ = self.vae(X_test)

    return X_recon.detach().cpu().numpy()


  def detach(self):
    self.vae = self.vae.to('cpu')


  def save_state(self):
    if args['vae']['save']:
      states_path = Path('./vae/vae_net.pth')
      torch.save(self.vae.state_dict(), states_path)
