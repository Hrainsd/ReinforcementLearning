import numpy as np
import random
import matplotlib.pyplot as plt

# Define the gridworld
gridworld = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, -1, -1, -1, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1, -1,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
alpha = 0.1
epsilon = 0.1
episodes = 50000
max_steps = 1000

# Define actions
actions = ['Up', 'Down', 'Left', 'Right']
action_effects = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

# Initialize Q-table
# Q = np.zeros((*gridworld.shape, len(actions)))
# Q值设置为0时，运算速度极慢，效果极差。零初始化可能导致初期学习较慢，因为智能体在学习过程中逐渐发现哪个动作更好。如果初始化的 Q-值过低，智能体可能需要更多的时间来探索并更新 Q-值。
Q = np.random.uniform(low=0.0, high=0.1, size=(*gridworld.shape, len(actions)))


# Epsilon-greedy policy
def epsilon_greedy_policy(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(Q[state])]


# Get next state
def get_next_state(state, action):
    row, col = state
    drow, dcol = action_effects[action]
    new_row, new_col = row + drow, col + dcol

    # Check boundaries
    if new_row < 0 or new_row >= gridworld.shape[0] or new_col < 0 or new_col >= gridworld.shape[1]:
        return state, rboundary

    # Check for forbidden states
    if gridworld[new_row, new_col] == rforbidden:
        return state, rboundary

    return (new_row, new_col), gridworld[new_row, new_col]


# Q-Learning algorithm
def q_learning(gridworld, Q, episodes, alpha, gamma, epsilon, max_steps):
    for episode in range(episodes):
        # state = (0,0)
        # Choose a random initial state that is not a terminal or forbidden state
        initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                          gridworld[i, j] == 0]
        state = random.choice(initial_states)

        for step in range(max_steps):
            action = epsilon_greedy_policy(state, Q, epsilon)
            next_state, reward = get_next_state(state, action)

            # Q-Learning update
            Q[state][actions.index(action)] += alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state][actions.index(action)])

            state = next_state

            if reward == rtarget:
                break

    return Q


# Training the agent
Q = q_learning(gridworld, Q, episodes, alpha, gamma, epsilon, max_steps)

# Print the Q-values
print("Q-values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            print(f"State ({i}, {j}): {Q[(i, j)]}")

# Extract the optimal policy
policy_grid = np.full(gridworld.shape, '', dtype=object)

for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            best_action = actions[np.argmax(Q[(i, j)])]
            policy_grid[i, j] = best_action.lower()

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)

# Compute state values from Q-values
def compute_state_values(Q):
    state_values = np.zeros(gridworld.shape)

    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:  # Only calculate for non-terminal states
                state = (i, j)
                q_values = Q[state]
                state_values[i, j] = np.mean(q_values)  # Average over all actions

    return state_values

# Compute and print state values
state_values = compute_state_values(Q)
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