import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random
import gym
from tqdm import trange
import matplotlib.pyplot as plt

# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

# -------------------------------------- #
# 深度学习网络
# -------------------------------------- #

class Net(nn.Module):
    def __init__(self, n_states, n_hidden1, n_hidden2, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------------- #
# 深度强化学习模型
# -------------------------------------- #

class DQN:
    def __init__(self, n_states, n_hidden1, n_hidden2, n_actions, learning_rate, gamma, epsilon, target_update, device):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0

        # 构建训练网络和目标网络
        self.q_net = Net(n_states, n_hidden1, n_hidden2, n_actions).to(self.device)
        self.target_q_net = Net(n_states, n_hidden1, n_hidden2, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.losses = []

    def take_action(self, state):
        state = torch.Tensor(state[np.newaxis, :]).to(self.device)
        if np.random.random() < self.epsilon:
            action = self.q_net(state).argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

# ------------------------------- #
# 主程序
# ------------------------------- #

# 设置设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 参数设置
capacity = 1000
lr = 1e-3
gamma = 0.99
epsilon = 0.9
target_update = 100
batch_size = 64
n_hidden1 = 256
n_hidden2 = 128
min_size = 200
num_episodes = 200

# 环境初始化
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# 经验回放池和DQN初始化
replay_buffer = ReplayBuffer(capacity)
agent = DQN(n_states, n_hidden1, n_hidden2, n_actions, lr, gamma, epsilon, target_update, device)

return_list = []
action_counts = np.zeros(n_actions)

# 训练过程
for i in trange(num_episodes, desc='Training'):
    state, _ = env.reset()
    episode_return = 0
    done = False

    while not done:
        action = agent.take_action(state)
        action_counts[action] += 1
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward

        if replay_buffer.size() > min_size:
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': s,
                'actions': a,
                'next_states': ns,
                'rewards': r,
                'dones': d,
            }
            agent.update(transition_dict)

    return_list.append(episode_return)

# 绘制回报曲线
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(range(len(return_list)), return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')

# 绘制平均回报曲线
average_returns = [np.mean(return_list[max(0, i-10):i+1]) for i in range(len(return_list))]
plt.subplot(3, 1, 2)
plt.plot(range(len(average_returns)), average_returns)
plt.xlabel('Episodes')
plt.ylabel('Average Returns')
plt.title('DQN Average Returns')

# 绘制损失曲线
plt.subplot(3, 1, 3)
plt.plot(range(len(agent.losses)), agent.losses)
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('DQN Training Loss')

plt.tight_layout()
plt.show()

# 绘制动作选择分布
plt.figure(figsize=(6, 4))
plt.bar(range(n_actions), action_counts)
plt.xlabel('Actions')
plt.ylabel('Counts')
plt.title('Action Distribution')
plt.show()
