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
alpha = 0.01
episodes = 10000
max_steps = 100

# Actions and their effects
actions = ['up', 'down', 'left', 'right']
action_effects = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Define policy network parameters
theta = np.random.uniform(low=0.0, high=0.01, size=(*gridworld.shape, len(actions)))

# Softmax policy
def softmax_policy(state):
    preferences = theta[state[0], state[1], :]
    exp_preferences = np.exp(preferences - np.max(preferences))
    action_probs = exp_preferences / np.sum(exp_preferences)
    return action_probs

# Get the next state
def get_next_state(state, action):
    row, col = state
    drow, dcol = action_effects[actions[action]]
    new_row, new_col = row + drow, col + dcol

    # Check boundaries
    if new_row < 0 or new_row >= gridworld.shape[0] or new_col < 0 or new_col >= gridworld.shape[1]:
        return state, rboundary

    # Check if forbidden
    if gridworld[new_row, new_col] == rforbidden:
        return state, rforbidden

    return (new_row, new_col), gridworld[new_row, new_col]

# Generate an episode
def generate_episode(max_steps):
    initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                      gridworld[i, j] == 0]
    state = random.choice(initial_states)
    episode = []

    for _ in range(max_steps):
        action_probs = softmax_policy(state)
        action = np.random.choice(len(actions), p=action_probs)
        next_state, reward = get_next_state(state, action)
        episode.append((state, action, reward))

        if reward == rtarget:  # Terminate on reaching target
            break

        state = next_state

    return episode

# Update policy (REINFORCE)
def update_policy(episode):
    global theta  # Make sure we update the global theta
    for t in range(len(episode)):
        state, action, reward = episode[t]

        # Calculate discounted return from time t onwards
        G = sum([gamma**(k - t) * episode[k][2] for k in range(t, len(episode))])

        # Calculate gradient
        action_probs = softmax_policy(state)
        grad_log_pi = np.zeros_like(action_probs)
        grad_log_pi[action] = 1.0 / action_probs[action]

        # Update policy parameters
        theta[state[0], state[1], :] += alpha * G * grad_log_pi

# Main loop
for episode_num in range(episodes):
    episode = generate_episode(max_steps)
    update_policy(episode)

    if episode_num % 100 == 0:
        print(f"Episode {episode_num}")

# Print Q-values
print("Q-values:")
Q_values = theta.copy()
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            print(f"State ({i}, {j}): {Q_values[i, j, :]}")


# Initialize policy grid
policy_grid = np.full(gridworld.shape, '', dtype=object)

# Populate the policy grid
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            action_values = theta[state[0], state[1], :]
            best_action = np.argmax(action_values)
            policy_grid[i, j] = actions[best_action].lower()

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)


# Calculate state value function
def compute_state_values():
    V = np.zeros(gridworld.shape)
    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:
                action_probs = softmax_policy((i, j))
                V[i, j] = np.dot(theta[i, j, :], action_probs)
    return V

# Print state value function
V = compute_state_values()
print("\nState Value Function:")
print(V)


# Plot state value and policy
fig, ax = plt.subplots()
cmap = plt.cm.Spectral
norm = plt.Normalize(vmin=V.min(), vmax=V.max())
cbar = ax.imshow(V, cmap=cmap, norm=norm, interpolation='nearest')

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

def monte_carlo_evaluation(policy, episodes):
    V = np.zeros(gridworld.shape)
    Q = np.zeros((*gridworld.shape, len(actions)))
    returns = {(i, j): [] for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1])}

    for _ in range(episodes):
        episode = generate_episode(max_steps)
        visited_state_actions = set()

        for t in range(len(episode)):
            state, action, reward = episode[t]
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                G = sum([gamma**(k - t) * episode[k][2] for k in range(t, len(episode))])
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                Q[state][action] = np.mean(returns[state])

    return V, Q

# Approximate V and Q using Monte Carlo evaluation
V, Q = monte_carlo_evaluation(softmax_policy, episodes)

print("Approximate Q-values:")
print(Q)

print("\nApproximate State Value Function:")
print(V)

# Plot the state value function
plt.imshow(V, cmap='winter', interpolation='none')
plt.colorbar(label='State Value')

# Annotate each grid cell with the value
for x in range(V.shape[0]):
    for y in range(V.shape[1]):
        plt.text(y, x, f'{V[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('MC State Value')
plt.show()
