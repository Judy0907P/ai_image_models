import torch
import wandb
from pathlib import Path

from ai_image_models.data import CIFAR10Dataset
from ai_image_models.models import FlowMLP
from ai_image_models.learner import Learner, train_classifier




def fid(f1, f2, eps=1e-6):
    # float64 + diagonal eps keeps eigvals numerically stable when features are near-degenerate
    f1, f2 = f1.double(), f2.double()
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = torch.cov(f1.T), torch.cov(f2.T)
    eye = torch.eye(c1.shape[0], dtype=c1.dtype)
    c1, c2 = c1 + eps * eye, c2 + eps * eye
    eig = torch.linalg.eigvals(c1 @ c2).real.clamp(min=0)
    return (((m1 - m2) ** 2).sum() + torch.trace(c1 + c2) - 2 * eig.sqrt().sum()).item()


def evaluate(learner, clf, ds):
  real = clf.features(ds.x[:2048])
  gen = clf.features(learner.generate(n=2048, steps=100))
  return fid(real, gen)


def main():
  # Defaults for a normal `python train.py` run.
  # Under a sweep, wandb replaces these with the YAML values after init.
  defaults = {
    "lr": 3e-4,
    "epochs": 10,
    "batch_size": 256,
  }
  # Do not hardcode project for sweeps — the agent already sets project/sweep.
  run = wandb.init(config=defaults)
  cfg = run.config
  print(f"config from wandb: lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}")

  data_dir = Path(__file__).resolve().parent / "../01_train/data/cifar10"
  ds = CIFAR10Dataset(data_dir)
  flow = FlowMLP(img_shape=(32, 32, 3))
  learner = Learner(flow, lr=cfg.lr)
  learner.learn(ds, epochs=cfg.epochs, batch_size=cfg.batch_size, logger=run)

  clf = train_classifier(ds, img_shape=(32, 32, 3), n_classes=10)
  run.summary["FID"] = evaluate(learner, clf, ds)
  run.finish()


if __name__ == "__main__":
  main()
