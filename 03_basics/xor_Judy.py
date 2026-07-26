import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import trange

X = torch.Tensor(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ]
)

Y = torch.Tensor(
    [
        [0],
        [1],
        [0],
        [1],
    ]
)


class NeuralNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer in [self.input_layer, self.hidden_layer, self.output_layer]:
            x = layer(x)
            x = torch.nn.functional.relu(x)
        return x


def loss_fn(y, y_hat):
    return torch.square(y - y_hat).mean()


def train(hidden_dim: int, epochs: int = 1000, lr: float = 1e-3):
    model = NeuralNet(hidden_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for _ in trange(epochs, desc=f"hidden={hidden_dim}"):
        epoch_loss = 0.0
        for x, y in zip(X, Y):
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(X))

    return losses, model


def main():
    hidden_dims = [64, 36, 4]
    epochs = 1000
    histories = {}

    for hidden_dim in hidden_dims:
        losses, model = train(hidden_dim, epochs=epochs)
        histories[hidden_dim] = losses
        print(f"\nhidden={hidden_dim} ({sum(p.numel() for p in model.parameters())} params)")
        for x in X:
            print(x.tolist(), model(x.unsqueeze(0)).item())

    styles = {
        64: {"color": "C0", "linestyle": "-", "marker": "o", "linewidth": 2},
        36: {"color": "C1", "linestyle": "--", "marker": "s", "linewidth": 2},
        4: {"color": "C2", "linestyle": "-.", "marker": "^", "linewidth": 2},
    }
    markevery = max(epochs // 20, 1)

    plt.figure(figsize=(8, 5))
    for hidden_dim, losses in histories.items():
        style = styles[hidden_dim]
        plt.plot(
            losses,
            label=f"hidden={hidden_dim}",
            markevery=markevery,
            markersize=5,
            **style,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.title("XOR training loss by hidden layer size")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "03_basics/xor_loss_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
