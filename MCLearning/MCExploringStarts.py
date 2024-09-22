import numpy as np
import random
import matplotlib.pyplot as plt

# Define the gridworld
gridworld = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, -1, -1, 0],
    [0, 0, -1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1]
])

# Define parameters
rboundary = -1
rforbidden = -1
rtarget = 1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-values and counters
Q_values = {s: {a: 0 for a in ['Up', 'Down', 'Left', 'Right']} for s in np.ndindex(gridworld.shape)}
returns = {s: {a: [] for a in ['Up', 'Down', 'Left', 'Right']} for s in np.ndindex(gridworld.shape)}

def get_possible_actions(state):
    x, y = state
    actions = []
    possible_actions = {
        'Up': (x - 1, y),
        'Down': (x + 1, y),
        'Left': (x, y - 1),
        'Right': (x, y + 1)
    }
    for action, (new_x, new_y) in possible_actions.items():
        if 0 <= new_x < gridworld.shape[0] and 0 <= new_y < gridworld.shape[1]:
            actions.append(action)
    return actions

def choose_action(state):
    # epsilon-greedy
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_possible_actions(state))  # Exploration
    else:
        # Greedy policy
        possible_actions = get_possible_actions(state)
        if not possible_actions:
            return None
        best_action = max(possible_actions, key=lambda a: Q_values[state][a])
        return best_action

def simulate_episode(start_state, start_action):
    state = start_state
    action = start_action
    episode = []
    while gridworld[state] != rboundary and gridworld[state] != rtarget:
        next_state = (state[0] + (1 if action == 'Down' else -1 if action == 'Up' else 0),
                      state[1] + (1 if action == 'Right' else -1 if action == 'Left' else 0))
        reward = gridworld[next_state]
        episode.append((state, action, reward))
        state = next_state
        action = choose_action(state)
        if action is None:
            break
    return episode

# Monte Carlo control with exploring starts using first-visit method
for episode in range(num_episodes):
    # Choose a random starting state that is not terminal
    while True:
        start_state = (random.randint(0, gridworld.shape[0] - 1), random.randint(0, gridworld.shape[1] - 1))
        if gridworld[start_state] == 0:
            break
    start_action = random.choice(get_possible_actions(start_state))
    episode = simulate_episode(start_state, start_action)

    # Track seen state-action pairs
    seen_state_actions = set()

    # Calculate returns and update Q-values using first-visit method
    G = 0
    for state, action, reward in reversed(episode):
        G = reward + gamma * G
        if (state, action) not in seen_state_actions:
            returns[state][action].append(G)
            Q_values[state][action] = np.mean(returns[state][action])
            seen_state_actions.add((state, action))


# Print state-action values
print("State-Action Values:")
for state in Q_values:
    print(f"State {state}: {Q_values[state]}")

# Initialize an empty grid for the policy
policy_grid = np.full(gridworld.shape, '', dtype=object)

# Determine and fill the policy grid and value grid
for state in np.ndindex(gridworld.shape):
    if gridworld[state] == rboundary or gridworld[state] == rtarget:
        policy_grid[state] = ''  # Empty space for terminal states
    else:
        best_action = max(Q_values[state], key=Q_values[state].get)
        policy_grid[state] = best_action.lower()  # Convert action to lowercas

# Print optimal policy
print("\nOptimal Policy:")
print(policy_grid)

# Compute state values from Q-values using the policy
def compute_state_values(Q_values, epsilon):
    state_values = np.zeros(gridworld.shape)

    for state in np.ndindex(gridworld.shape):
        possible_actions = get_possible_actions(state)
        if possible_actions:
            # Epsilon-greedy policy: calculate action probabilities
            action_probs = np.zeros(len(possible_actions))
            best_action = choose_action(state)
            for i, action in enumerate(possible_actions):
                if action == best_action:
                    action_probs[i] = 1 - epsilon + (epsilon / len(possible_actions))
                else:
                    action_probs[i] = epsilon / len(possible_actions)

            # Compute state value as the weighted sum of Q-values
            state_values[state] = sum(Q_values[state][a] * action_probs[i]
                                      for i, a in enumerate(possible_actions))

    return state_values

# Compute and print state values
state_values = compute_state_values(Q_values, epsilon)
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
        if gridworld[x, y] != rforbidden and gridworld[
            x, y] != rtarget:  # Annotate only non-forbidden and non-target states
            plt.text(y, x, f'{state_values[x, y]:.2f}', ha='center', va='center', color='yellow')

plt.title('State Value')
plt.show()
