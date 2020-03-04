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
        self.fc1 = torch.nn.Linear(30 * 10 * 7, 64)
        self.fc2 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        y = self.pool(F.relu(self.conv2(x)))
        z = (self.fc1(y.view(-1, 30 * 10 * 7)))
        return torch.sigmoid(self.fc2(z))

    def loss(self, y, y_pred):
        F.mse_loss()


fig, axs = plt.subplots(nrows=1, ncols=3)
m = Model().float()
env = gym.make('Breakout-v0')
env.reset()
prev_state = env.env.ale.getScreenRGB()
out = m.forward(torch.tensor(prev_state).view(1, 3, 210, 160).float())
J=

J = m.loss()
print(out)
