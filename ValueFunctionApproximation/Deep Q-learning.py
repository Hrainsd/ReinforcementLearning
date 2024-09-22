import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
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
alpha = 0.01
epsilon = 0.1
episodes = 100
batch_size = 32
target_update_freq = 10
max_steps = 100  # Maximum steps to prevent infinite loops

# Define possible actions
actions = ['up', 'down', 'left', 'right']
action_size = len(actions)

# Action effects
action_effects = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Epsilon-greedy policy
def epsilon_greedy_policy(state, model, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        q_values = model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

# Get next state
def get_next_state(state, action):
    row, col = state
    drow, dcol = action_effects[actions[action]]
    new_row, new_col = row + drow, col + dcol

    # Check boundaries
    if new_row < 0 or new_row >= gridworld.shape[0] or new_col < 0 or new_col >= gridworld.shape[1]:
        return state, rboundary

    # Check for forbidden states
    if gridworld[new_row, new_col] == rforbidden:
        return state, rboundary

    return (new_row, new_col), gridworld[new_row, new_col]

# Build DQN model
def build_dqn(input_shape, output_size):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_size)
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=alpha))
    return model

# Initialize replay buffer
def create_replay_buffer(max_size):
    return deque(maxlen=max_size)

def add_to_replay_buffer(buffer, experience):
    buffer.append(experience)

def sample_from_replay_buffer(buffer, batch_size):
    return random.sample(buffer, batch_size)

def replay_buffer_size(buffer):
    return len(buffer)

# Train DQN
def train_dqn(model, target_model, buffer, episodes, epsilon):
    for episode in range(episodes):
        initial_states = [(i, j) for i in range(gridworld.shape[0]) for j in range(gridworld.shape[1]) if
                          gridworld[i, j] == 0]
        state = random.choice(initial_states)

        total_reward = 0
        steps = 0

        while True:
            action = epsilon_greedy_policy(state, model, epsilon)
            next_state, reward = get_next_state(state, action)
            total_reward += reward
            add_to_replay_buffer(buffer, (state, action, reward, next_state, reward))
            state = next_state
            steps += 1

            if gridworld[state[0], state[1]] == rtarget or steps >= max_steps:
                break

            if replay_buffer_size(buffer) >= batch_size:
                minibatch = sample_from_replay_buffer(buffer, batch_size)
                states, actions, rewards, next_states, _ = zip(*minibatch)

                q_values_next = target_model.predict(np.array(next_states))
                targets = np.array(rewards) + gamma * np.max(q_values_next, axis=1)

                q_values = model.predict(np.array(states))
                for i in range(batch_size):
                    q_values[i][actions[i]] = targets[i]

                model.fit(np.array(states), q_values, epochs=5, verbose=1)

                if episode % target_update_freq == 0:
                    target_model.set_weights(model.get_weights())

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return model

# Initialize models and replay buffer
input_shape = (2,)  # State representation (change as needed)
model = build_dqn(input_shape, action_size)
target_model = build_dqn(input_shape, action_size)
target_model.set_weights(model.get_weights())
buffer = create_replay_buffer(max_size=10000)

# Train the model
trained_model = train_dqn(model, target_model, buffer, episodes, epsilon)

# Print the Q-values
print("\nQ-values:")
for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            state_onehot = np.expand_dims(state, axis=0)
            q_values = trained_model.predict(state_onehot, verbose=0)[0]
            print(f"State {state}: {q_values}")


# Extract the optimal policy
policy_grid = np.full(gridworld.shape, '', dtype=object)

for i in range(gridworld.shape[0]):
    for j in range(gridworld.shape[1]):
        if gridworld[i, j] == 0:
            state = (i, j)
            q_values = trained_model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            best_action = np.argmax(q_values)
            policy_grid[i, j] = actions[best_action]

# Print the optimal policy
print("\nOptimal Policy:")
print(policy_grid)

# Compute state values from Q-values
def compute_state_values(model):
    state_values = np.zeros(gridworld.shape)

    for i in range(gridworld.shape[0]):
        for j in range(gridworld.shape[1]):
            if gridworld[i, j] == 0:  # Only calculate for non-terminal states
                state = np.array([i, j])
                q_values = model.predict(np.expand_dims(state, axis=0))[0]
                state_values[i, j] = np.mean(q_values)  # Average over all actions

    return state_values

# Compute and print state values
state_values = compute_state_values(trained_model)
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
