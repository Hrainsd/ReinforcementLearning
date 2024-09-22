import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Define the gridworld
gridworld = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, -1, -1, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, -1, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
alpha_actor = 0.01  # Learning rate for actor
alpha_critic = 0.01  # Learning rate for critic
tau = 0.005  # Target network update rate
episodes = 10000
max_steps = 100  # Maximum steps per episode
memory_size = 1000  # Size of replay buffer
batch_size = 64  # Size of mini-batch from replay buffer

# Define actions
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def size(self):
        return len(self.buffer)


# Function to choose an action using the actor network with added noise
def choose_action(state, theta_actor, noise_scale=0.1):
    action_values = theta_actor[state[0], state[1], :]
    action_index = np.argmax(action_values + np.random.normal(0, noise_scale, size=num_actions))
    return actions[action_index]


# Function to take an action and return the next state and reward
def take_action(state, action):
    row, col = state
    new_row, new_col = row, col

    if action == 'up':
        new_row = max(row - 1, 0)
    elif action == 'down':
        new_row = min(row + 1, gridworld.shape[0] - 1)
    elif action == 'left':
        new_col = max(col - 1, 0)
    elif action == 'right':
        new_col = min(col + 1, gridworld.shape[1] - 1)

    if gridworld[new_row, new_col] == rforbidden:
        new_row, new_col = row, col

    reward = gridworld[new_row, new_col]
    return (new_row, new_col), reward


# Initialize actor and critic networks and their targets
theta_actor = np.zeros((gridworld.shape[0], gridworld.shape[1], num_actions))
theta_critic = np.zeros((gridworld.shape[0], gridworld.shape[1], num_actions))
theta_actor_target = np.copy(theta_actor)
theta_critic_target = np.copy(theta_critic)

# Initialize replay buffer
replay_buffer = ReplayBuffer(memory_size)

# DDPG algorithm
for episode in range(episodes):
    initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                      gridworld[i, j] == 0]
    state = random.choice(initial_states)

    for step in range(max_steps):
        # Choose action using actor network with added noise
        action = choose_action(state, theta_actor)

        # Take action and observe next state and reward
        next_state, reward = take_action(state, action)

        # Store transition in replay buffer
        replay_buffer.store((state, actions.index(action), reward, next_state))

        # Sample from replay buffer
        if replay_buffer.size() > batch_size:
            states, actions_batch, rewards, next_states = replay_buffer.sample(batch_size)

            # Update critic network
            for i in range(batch_size):
                next_action = choose_action(next_states[i], theta_actor_target)
                target = rewards[i] + gamma * theta_critic_target[
                    next_states[i][0], next_states[i][1], actions.index(next_action)]
                td_error = target - theta_critic[states[i][0], states[i][1], actions_batch[i]]
                theta_critic[states[i][0], states[i][1], actions_batch[i]] += alpha_critic * td_error

            # Update actor network
            for i in range(batch_size):
                action_probs = np.exp(theta_actor[states[i][0], states[i][1], :]) / np.sum(
                    np.exp(theta_actor[states[i][0], states[i][1], :]))
                grad_actor = (1 - action_probs[actions_batch[i]]) * theta_critic[
                    states[i][0], states[i][1], actions_batch[i]]
                theta_actor[states[i][0], states[i][1], actions_batch[i]] += alpha_actor * grad_actor

            # Update target networks
            theta_actor_target = tau * theta_actor + (1 - tau) * theta_actor_target
            theta_critic_target = tau * theta_critic + (1 - tau) * theta_critic_target

        if reward == rtarget:
            break
        state = next_state

# Compute Q-values from critic network
Q = np.zeros((gridworld.shape[0], gridworld.shape[1], num_actions))
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            for a in range(num_actions):
                Q[i, j, a] = theta_critic[i, j, a]  # Q-values are directly from the critic network

# Print the Q-values
print("\nQ-values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            q_values = Q[i, j]
            print(f"State ({i}, {j}): {q_values}")

# Extract the optimal policy
policy_grid = np.full(gridworld.shape, '', dtype=object)

for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            action_values = Q[i, j]
            best_action = np.argmax(action_values)
            policy_grid[i, j] = actions[best_action]

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)


# Function to compute state values from Q-values
def compute_state_values(theta_actor, theta_critic):
    state_values = np.zeros(gridworld.shape)

    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:  # Only calculate for non-terminal states
                action_probs = np.exp(theta_actor[i, j, :]) / np.sum(np.exp(theta_actor[i, j, :]))
                state_values[i, j] = np.sum(action_probs * theta_critic[i, j, :])

    return state_values


# Compute state values
state_values = compute_state_values(theta_actor, theta_critic)

# Print state values
print("\nState Values:")
print(state_values)

# Plot state value and policy
fig, ax = plt.subplots()
cmap = plt.cm.Spectral
norm = plt.Normalize(vmin=state_values.min(), vmax=state_values.max())
cbar = ax.imshow(state_values, cmap=cmap, norm=norm, interpolation='nearest')

# Add the policy arrows
for i in range(policy_grid.shape[0]):
    for j in range(policy_grid.shape[1]):
        if policy_grid[i, j]:
            if policy_grid[i, j] == 'up':
                ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'down':
                ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'left':
                ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')
            elif policy_grid[i, j] == 'right':
                ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.2, fc='k', ec='k')

# Display grid and labels
ax.set_xticks(np.arange(gridworld.shape[1]))
ax.set_yticks(np.arange(gridworld.shape[0]))
ax.set_xticklabels(np.arange(1, gridworld.shape[1] + 1))
ax.set_yticklabels(np.arange(1, gridworld.shape[0] + 1))
ax.grid(which='both', color='black', linestyle='-', linewidth=2)

# Set terminal state labels and colors
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == -1:
            ax.text(j, i, 'T', va='center', ha='center', color='white')
        elif gridworld[i, j] == 1:
            ax.text(j, i, 'G', va='center', ha='center', color='white')

plt.title('Policy Visualization')
plt.colorbar(cbar, ax=ax, label='State Value')
plt.show()

# Plot state values
plt.imshow(state_values, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(state_values.shape[0]):
    for y in range(state_values.shape[1]):
        if gridworld[x, y] == 0:  # Only annotate non-terminal states
            plt.text(y, x, f'{state_values[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()
