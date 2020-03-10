import gym
import torch
import torch.nn.functional as F
import collections
import numpy as np

torch.manual_seed(42)


class ReplayMemory:

    def __init__(self, max_size):
        self.mem = collections.deque(maxlen=max_size)

    def sample(self, size):
        length = len(self.mem)
        samp = np.random.choice(np.arange(size), size=size, replace=False)
        print(samp)
        return [self.mem[index] for index in samp]

    def add(self, x):
        self.mem.append(x)

    def __setitem__(self, key, value):
        self.mem[key] = value


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 30, 10)
        self.pool = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = torch.nn.Conv2d(30, 30, 10)
        self.fc1 = torch.nn.Linear(30 * 10 * 7, 64)
        self.fc2 = torch.nn.Linear(64, 4)

    def forward(self, x):
        self.convout1 = self.pool(F.relu(self.conv1(x)))
        self.convout2 = self.pool(F.relu(self.conv2(self.convout1)))
        z = (self.fc1(self.convout2.view(-1, 30 * 10 * 7)))
        return torch.sigmoid(self.fc2(z))

    # def loss(self, y, y_pred):
    #     F.mse_loss()


# initialize function approximator and target models
function_m = Model()
target_m = Model()
# initialize replay memory with capacity 1000
memory_size = 1000
replay_memory = ReplayMemory(memory_size)

episodes = 3000
batch_size = 100
discount = 0.1

env = gym.make('Breakout-v0')
env.reset()
# initialize Replay Memory and populate it
for i in range(memory_size):
    prev_state = env.env.ale.getScreenRGB()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    replay_memory.add((prev_state, action, reward, done, next_state))
done = False
i = 99
j = 0
for ep in range(episodes):
    while not done:
        prev_state = env.env.ale.getScreenRGB()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        replay_memory[i % 1000] = (prev_state, action, reward, done, next_state)
        i += 1

        # every 100 steps, take a gradient descent step
        if i % 100 == 0:
            batch = replay_memory.sample(batch_size)
            for sample in batch:
                input_, action_, reward_, done_, next_state_ = sample
                preds = function_m.forward(torch.tensor(input_).view(1, 3, 210, 160).float())
                next_qs = target_m.forward(torch.tensor(next_state_).view(1, 3, 210, 160).float())
                if done_:
                    lab = reward_
                else:
                    maxval, argmax = torch.max(next_qs, -1)
                    lab = reward_ + discount * argmax
                    labels=[]
                    # TODO: set all labels same as pred except for argmax, where it is set to lab
                loss = F.mse_loss(preds, labels)
                loss.backward()
            with torch.no_grad():
                pass
        # every thousand steps, update the target network weights
        if i % 1000 == 0:
            with torch.no_grad():
                target_m[0].conv1.weight = function_m[0].conv1.weight
                target_m[0].conv2.weight = function_m[0].conv2.weight
                target_m[0].fc1.weight = function_m[0].fc1.weight
                target_m[0].fc2.weight = function_m[0].fc2.weight

    env.reset()
    # Perform a single step of gradient descent
