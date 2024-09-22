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
episodes = 10000
max_steps = 100

# Define actions
actions = ['Up', 'Down', 'Left', 'Right']
action_effects = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

# Feature function
def feature_vector(state, action):
    feature = np.zeros(gridworld.size * len(actions))
    index = (state[0] * gridworld.shape[1] + state[1]) * len(actions) + actions.index(action)
    feature[index] = 1
    return feature

# Initialize weights
weights = np.zeros(gridworld.size * len(actions))

# Q-value function approximation
def Q_value(state, action, weights):
    return np.dot(weights, feature_vector(state, action))

# Epsilon-greedy policy
def epsilon_greedy_policy(state, weights, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        q_values = [Q_value(state, action, weights) for action in actions]
        return actions[np.argmax(q_values)]

# Get next state
def get_next_state(state, action):
    row, col = state
    drow, dcol = action_effects[action]
    new_row, new_col = row + drow, col + dcol
    if new_row < 0 or new_row >= gridworld.shape[0] or new_col < 0 or new_col >= gridworld.shape[1] or gridworld[
        new_row, new_col] == -1:
        return state, rboundary
    else:
        return (new_row, new_col), gridworld[new_row, new_col]

# Sarsa with function approximation
def sarsa_function_approx(gridworld, weights, episodes, alpha, gamma, epsilon, max_steps):
    for episode in range(episodes):
        # Choose a random initial state that is not a terminal or forbidden state
        initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                          gridworld[i, j] == 0]
        state = random.choice(initial_states)
        action = epsilon_greedy_policy(state, weights, epsilon)

        for step in range(max_steps):
            next_state, reward = get_next_state(state, action)
            next_action = epsilon_greedy_policy(next_state, weights, epsilon)

            # Calculate the TD error
            td_target = reward + gamma * Q_value(next_state, next_action, weights)
            td_error = td_target - Q_value(state, action, weights)

            # Update weights
            weights += alpha * td_error * feature_vector(state, action)

            state = next_state
            action = next_action

            if reward == rtarget:
                break

    return weights

# Training the agent
weights = sarsa_function_approx(gridworld, weights, episodes, alpha, gamma, epsilon, max_steps)

# Print the Q-Values
print("Q-Values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            print(f"State ({i}, {j}):")
            for action in actions:
                q_value = Q_value((i, j), action, weights)
                print(f"  {action}: {q_value:.4f}")

# Extract the optimal policy
policy_grid = np.full(gridworld.shape, '', dtype=object)

for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            q_values = [Q_value((i, j), action, weights) for action in actions]
            best_action = actions[np.argmax(q_values)]
            policy_grid[i, j] = best_action.lower()

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)

# Compute state values from Q-values using function approximation
def compute_state_values(weights, epsilon):
    state_values = np.zeros(gridworld.shape)

    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:  # Only calculate for non-terminal states
                state = (i, j)
                q_values = [Q_value(state, action, weights) for action in actions]
                # Compute the policy for the state
                action_probabilities = np.zeros(len(actions))
                for action in actions:
                    if random.uniform(0, 1) < epsilon:
                        action_probabilities[actions.index(action)] = 1.0 / len(actions)
                    else:
                        action_probabilities[actions.index(action)] = np.exp(Q_value(state, action, weights)) / np.sum(
                            np.exp(q_values))
                state_values[i, j] = np.dot(action_probabilities, q_values)  # Weighted average of Q-values

    return state_values

# Compute and print state values
state_values = compute_state_values(weights, epsilon)
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
