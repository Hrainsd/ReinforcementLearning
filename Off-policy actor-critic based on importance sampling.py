import numpy as np
import random
import matplotlib.pyplot as plt

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
alpha = 0.01  # Learning rate for actor
beta = 0.01   # Learning rate for critic
epsilon = 0.1
episodes = 50000
max_steps = 100  # Maximum steps per episode

# Define actions
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)

# Function to choose an action using softmax policy
def choose_action(state, theta):
    probs = np.exp(theta[state[0], state[1], :]) / np.sum(np.exp(theta[state[0], state[1], :]))
    action_index = np.random.choice(num_actions, p=probs)
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

# Initialize policy parameters and value function
theta = np.zeros((gridworld.shape[0], gridworld.shape[1], num_actions))
values = np.zeros(gridworld.shape)

# Off-policy Actor-Critic with Importance Sampling
for episode in range(episodes):
    initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                      gridworld[i, j] == 0]
    state = random.choice(initial_states)
    trajectory = []

    for step in range(max_steps):
        # Choose action using behavior policy (epsilon-greedy in this case)
        if np.random.rand() < epsilon:
            action = actions[np.random.randint(num_actions)]
        else:
            action = choose_action(state, theta)

        next_state, reward = take_action(state, action)
        trajectory.append((state, action, reward))

        if reward == rtarget:
            break
        state = next_state

    # Calculate returns and importance sampling ratios
    G = 0
    W = 1
    returns = []
    for t in range(len(trajectory) - 1, -1, -1):
        state_t, action_t, reward_t = trajectory[t]

        G = gamma * G + reward_t
        returns.append(G)

        # Importance sampling ratio
        prob_behavior = epsilon / num_actions + (1 - epsilon) * (
            np.exp(theta[state_t[0], state_t[1], actions.index(action_t)]) / np.sum(
                np.exp(theta[state_t[0], state_t[1], :]))
        )
        prob_target = np.exp(theta[state_t[0], state_t[1], actions.index(action_t)]) / np.sum(
            np.exp(theta[state_t[0], state_t[1], :])
        )
        W = W * (prob_target / prob_behavior)

        # Update value function
        values[state_t] += beta * W * (G - values[state_t])

        # Update policy parameters
        delta_theta = alpha * W * (G - values[state_t]) * (
            np.eye(num_actions)[actions.index(action_t)] - np.exp(
                theta[state_t[0], state_t[1], :]) / np.sum(
                np.exp(theta[state_t[0], state_t[1], :]))
        )
        theta[state_t[0], state_t[1], :] += delta_theta

# Compute Q-values from policy parameters
Q = np.zeros((gridworld.shape[0], gridworld.shape[1], num_actions))
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            for a in range(num_actions):
                next_state, reward = take_action((i, j), actions[a])
                Q[i, j, a] = reward + gamma * values[next_state]

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

# Calculate state values based on the policy
def compute_state_values(theta):
    state_values = np.zeros(gridworld.shape)

    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:  # Only calculate for non-terminal states
                action_probs = np.exp(theta[i, j, :]) / np.sum(np.exp(theta[i, j, :]))
                state_values[i, j] = np.sum(action_probs * np.max(theta[i, j, :]))

    return state_values

# Compute state values
state_values = compute_state_values(theta)

# Print state values
print("\nState Values:")
print(state_values)


# Plot Q-values and policy
fig, ax = plt.subplots()
cmap = plt.cm.Spectral
norm = plt.Normalize(vmin=state_values.min(), vmax=state_values.max())
cbar = ax.imshow(vstate_values, cmap=cmap, norm=norm, interpolation='nearest')

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
