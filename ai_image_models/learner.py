import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from ai_image_models.models import SpriteClassifier


def sample_t(n, device):
    return torch.sigmoid(torch.randn(n, device=device))


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def add_noise(x, t, e):
    t = append_dims(t, x.ndim)
    return t * x + (1 - t) * e


def velocity(x, z, t):
    return (x - z) / (1 - append_dims(t, x.ndim)).clamp(min=0.05)


def flow_loss(model, x):
    e = torch.randn_like(x)
    t = sample_t(x.shape[0], x.device)
    z = add_noise(x, t, e)
    v = velocity(x, z, t)
    v_pred = velocity(model(z, t), z, t)
    return (v - v_pred).pow(2).mean()


@torch.no_grad()
def sample(model, n, img_shape, steps=300, device="cpu", trajectory=False):
    z = torch.randn(n, *img_shape, device=device)
    dt = 1.0 / steps
    frames = []
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        z = z + velocity(model(z, t), z, t) * dt
        if trajectory:
            frames.append(z.clone())
    return torch.stack(frames) if trajectory else z


def train_classifier(dataset, epochs=5, batch_size=256, lr=1e-3, device=None,img_shape=(16, 16, 3),n_classes=5):
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = SpriteClassifier(img_shape=img_shape, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    bar = trange(epochs)
    for epoch in bar:
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.argmax(1).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        bar.set_description(f"epoch {epoch:02d} | acc {correct / total:.4f}")
    return model


class Learner:
    def __init__(self, model, device=None, lr=3e-4):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def learn(self, dataset, epochs=50, batch_size=256, logger=None):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.model.train()
        bar = trange(epochs)
        for epoch in bar:
            total = 0.0
            for x, _ in loader:
                x = x.to(self.device)
                loss = flow_loss(self.model, x)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total += loss.item()
                if logger:
                  logger.log({'loss': loss.item()})
            bar.set_description(f"epoch {epoch:02d} | loss {total / len(loader):.4f}")

    @torch.no_grad()
    def generate(self, n=64, steps=300, trajectory=False):
        self.model.eval()
        out = sample(self.model, n, self.model.img_shape, steps, self.device, trajectory)
        return out.clamp(0, 1).cpu()
