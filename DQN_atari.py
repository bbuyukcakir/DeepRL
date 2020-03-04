import numpy as np
import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Conv2d

torch.manual_seed(42)


class Model(torch.nn.Module):
    conv1: Conv2d

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, 10)
        self.pool = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = torch.nn.Conv2d(30, 30, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        y = self.pool(x)
        y = F.relu(self.conv2(y))
        res = self.pool(y)
        return y, res


fig, axs = plt.subplots(nrows=1, ncols=3)
m = Model()
env = gym.make('Breakout-v0')
env.reset()
observation, reward, done, info = env.step(1)
obs = torch.tensor(observation/255)
# a = (m.conv1(obs.view(1, 3, 210, 160)))
# x, y = m.forward(torch.tensor(observation).view(1, 3, 210, 160))
#
# axs[0].imshow(observation)
# axs[1].imshow(x)
# axs[2].imshow(y)
