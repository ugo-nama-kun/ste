import numpy as np
import torch

import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from nist import NistHandle

# Data: 64 (8x8) nist images
nh = NistHandle(flat=True)


# Model
class StochasticModel(nn.Module):
    def __init__(self):
        super(StochasticModel, self).__init__()

        self.fc1 = nn.Linear(2, 1)
        self.fc2 = nn.Linear(1, 100)
        self.fc3 = nn.Linear(100, 64)

    def forward(self, x):
        h = F.one_hot(x, num_classes=2).float()
        h = self.fc1(h)
        p = torch.sigmoid_(h)

        dist = Bernoulli(probs=p)
        z = dist.sample()
        h_ = z.detach() + p - p.detach()  # straight-through gradient computation

        h = torch.relu_(self.fc2(h_))
        y = torch.sigmoid_(self.fc3(h))
        return y, p, dist.entropy()


def get_image(model):
    n_img = 16

    y0, p0, _ = model(torch.zeros((n_img,), dtype=torch.long))
    y1, p1, _ = model(torch.ones((n_img,), dtype=torch.long))

    y0 = y0.detach().numpy()
    y1 = y1.detach().numpy()

    print(f"   x=0: p={p0.detach().mean()}, x=1: p={p1.detach().mean()}")

    im0 = np.hstack([y0[i].reshape(8, 8) for i in range(n_img)])
    im1 = np.hstack([y1[i].reshape(8, 8) for i in range(n_img)])

    plt.clf()
    plt.subplot(211)
    plt.title("$x = 0$")
    plt.imshow(im0, "gray")

    plt.subplot(212)
    plt.title("$x = 1$")
    plt.imshow(im1, "gray")

    plt.pause(0.0001)


def train():
    plt.figure()

    model = StochasticModel()
    optimizer = torch.optim.Adam(model.parameters())
    loss_mse = nn.MSELoss()

    n_iteration = 5000
    mini_batch_size = 32

    for n in range(n_iteration):
        total_loss = 0

        for _ in range(10):
            # Data
            x = torch.randint(0, 2, (mini_batch_size,))
            y_t = np.zeros((mini_batch_size, 64))

            for i in range(mini_batch_size):
                r = np.random.rand()

                if x[i] == 0:
                    if r < 0.4:
                        img = nh.get(target=3)
                    else:
                        img = nh.get(target=4)
                else:
                    if r < 0.6:
                        img = nh.get(target=3)
                    else:
                        img = nh.get(target=4)

                y_t[i] = img

            y_t = torch.as_tensor(y_t / 16., dtype=torch.float32)

            y_pred, p_pred, entropy = model(x)

            loss = loss_mse(y_pred, y_t) - 0.01 * entropy.mean()

            total_loss += loss.item()

            # training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if n % 20 == 0:
            print(f"{n}-th iteration: LOSS {total_loss}: Entropy: {entropy.mean().item()}")
            get_image(model)

    print("Finish")
    plt.show()


if __name__ == '__main__':
    # dist = Bernoulli(probs=torch.tensor([0.5]))
    # for i in range(10):
    #     print(dist.sample(sample_shape=(4, 1)))

    train()
