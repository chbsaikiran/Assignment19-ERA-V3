import numpy as np

# GridWorld size and parameters
grid_size = 4
num_states = grid_size * grid_size
gamma = 1.0
theta = 1e-4  # Convergence threshold

# Rewards: -1 per move, 0 for terminal state
rewards = -1 * np.ones(num_states)
rewards[-1] = 0  # Terminal state

# Value function initialization
V = np.zeros(num_states)

# Possible actions: up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row_offset, col_offset)

# Function to get next state index from (row, col) position
def get_state(row, col):
    return row * grid_size + col

# Function to get (row, col) from state index
def get_position(state):
    return divmod(state, grid_size)

# Function to get valid next states for a given state
def get_next_states(state):
    row, col = get_position(state)
    next_states = []
    for dr, dc in actions:
        new_row = row + dr
        new_col = col + dc
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            next_state = get_state(new_row, new_col)
        else:
            next_state = state  # Hitting wall stays in same state
        next_states.append(next_state)
    return next_states

# Value Iteration
iteration = 0
while True:
    delta = 0
    V_new = np.copy(V)
    for s in range(num_states):
        if s == num_states - 1:
            continue  # Terminal state
        next_states = get_next_states(s)
        expected_value = 0
        for s_prime in next_states:
            expected_value += 0.25 * (rewards[s_prime] + gamma * V[s_prime])
        delta = max(delta, abs(expected_value - V[s]))
        V_new[s] = expected_value
    V = V_new
    iteration += 1
    if delta < theta:
        break

# Print final value function
print("Converged in", iteration, "iterations.")
print("Value Function:")
print(V.reshape((grid_size, grid_size)))
