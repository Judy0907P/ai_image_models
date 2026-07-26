import torch
from torch import nn
import wandb
import einops
from pathlib import Path

from ai_image_models.data import CIFAR10Dataset
from ai_image_models.models import FlowMLP, time_embed
from ai_image_models.learner import Learner, train_classifier
from ai_image_models.eval import fid

class FlowCNN(nn.Module):
    def __init__(self, img_shape=(16, 16, 3), t_dim=128, h=128):
        super().__init__()
        self.img_shape = img_shape
        self.t_dim = t_dim

        # Early spatial features from image only (no huge channel concat).
        self.early = nn.Sequential(
            nn.Conv2d(3, h, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        )
        # Project time embed -> h channels, then add onto feature maps.
        self.t_proj = nn.Linear(t_dim, h)
        self.late = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            nn.Conv2d(h, 3, kernel_size=3, stride=1, padding=1), nn.SiLU(),
        )

    def forward(self, z, t):
        z = einops.rearrange(z, 'b h w c -> b c h w')
        h = self.early(z)
        # Inject time mid-network: broadcast (B, h) -> (B, h, 1, 1) and add.
        h = h + self.t_proj(time_embed(t, self.t_dim))[:, :, None, None]
        h = self.late(h)
        return einops.rearrange(h, 'b c h w -> b h w c')



def evaluate(learner, clf, ds):
  real = clf.features(ds.x[:2048])
  gen = clf.features(learner.generate(n=2048, steps=100))
  return fid(real, gen)


def main():
  config = {
   'lr': 1e-4,
   'epochs': 1,
   'batch_size': 256,
   'arch': 'cnn',
   'hidden_dim': 128
  }
  run = wandb.init(project="ai_image_model", config=config)
  cfg = run.config

  data_dir = Path(__file__).resolve().parent / "../01_train/data/cifar10"
  ds = CIFAR10Dataset(data_dir)
  if cfg.arch == 'mlp':
    model = FlowMLP(img_shape=(32, 32, 3), h=cfg.hidden_dim)
  elif cfg.arch == 'cnn':
    model = FlowCNN(img_shape=(32, 32, 3), h=cfg.hidden_dim)

  learner = Learner(model, lr=cfg.lr)
  learner.learn(ds, epochs=cfg.epochs, batch_size=cfg.batch_size, logger=run)

  
  clf = train_classifier(ds, img_shape=(32, 32, 3), n_classes=10)
  run.summary['FID'] = evaluate(learner, clf, ds)


if __name__ == "__main__":
  main()
