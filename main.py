import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.distributions import Bernoulli

from nist import NistHandle

# Data: 64 (8x8) nist images
nh = NistHandle(flat=True)


# Model
class StochasticModel(nn.Module):
    def __init__(self):
        super(StochasticModel, self).__init__()

        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 64)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.sigmoid_(h)

        dist = Bernoulli(probs=h)
        z = dist.sample()

        h_ = z.detach() + h - h.detach()
        y = torch.sigmoid_(self.fc2(h_))
        return y


def get_image(model):
    n_img = 16

    y0 = model(torch.zeros((n_img, 1))).numpy()
    y1 = model(torch.ones((n_img, 1))).numpy()

    im0 = np.vstack([y0[i].reshape(8, 8) for i in range(n_img)])
    im1 = np.vstack([y0[i].reshape(8, 8) for i in range(n_img)])

    plt.clf()
    plt.subplot(211)
    plt.imshow(im0, "gray")
    plt.subplot(212)
    plt.imshow(im1, "gray")
    plt.pause(3)


def train():
    fig = plt.figure()

    model = StochasticModel()
    optimizer = torch.optim.Adam(model.parameters())
    loss_mse = nn.MSELoss()

    n_iteration = 30
    mini_batch_size = 32

    for n in range(n_iteration):
        total_loss = 0

        for _ in range(6):
            # Data
            x = torch.randint(0, 2, (mini_batch_size, 1))
            y_t = np.zeros(mini_batch_size, 64)

            for i in range(mini_batch_size):
                r = np.random.rand()

                if x[i] == 0:
                    if r < 0.8:
                        img = nh.get(target=0)
                    else:
                        img = nh.get(target=1)
                else:
                    if r < 0.2:
                        img = nh.get(target=0)
                    else:
                        img = nh.get(target=1)

                y_t[i] = img

            y_t = torch.as_tensor(y_t / 16., dtype=torch.float32)

            y_predict = model(x)

            loss = loss_mse(y_predict, y_t)

            total_loss += loss.item()

            # training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{n}-th iteration: LOSS {total_loss}")
        get_image(model)

    print("Finish")


if __name__ == '__main__':
    dist = Bernoulli(probs=[0.5])
    for i in range(10):
        print(dist.sample(sample_shape=(4,)))