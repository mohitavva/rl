# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

# %%
def prepare_grid(grid_size, obstacle_ratio):
    grid = np.zeros((grid_size, grid_size))
    num_obstacles = int(grid_size*grid_size*obstacle_ratio)

    for i in range(num_obstacles):
        x, y = np.random.randint(0, grid_size, size=2)
        grid[x, y] = -1

    start = (random.randint(0, grid_size-1), random.randint(0, grid_size - 1))
    goal = (random.randint(0, grid_size-1), random.randint(0, grid_size - 1))

    while ((grid[start]) == -1 or grid[goal] == -1 or start == goal):
        start = (random.randint(0, grid_size-1), random.randint(0, grid_size - 1))
        goal = (random.randint(0, grid_size-1), random.randint(0, grid_size - 1))

    grid[start] = 2
    grid[goal] = 3

    return grid, start, goal

def reward_grid(grid, grid_size, reward_def, reward_goal, reward_obstacle):
    rewards = np.full((grid_size, grid_size), reward_def)
    rewards[grid == 3] = reward_goal
    rewards[grid == -1] = reward_obstacle

    return rewards
    

# %%
grid_size = 25
obstacle_ratio = 0.25
actions = ["up", "down", "left", "right"]
reward_def = -1
reward_goal = 100
reward_obstacle = -10
threshold = 1e-5
gamma = 0.8


grid, start, goal = prepare_grid(grid_size, obstacle_ratio)
plt.title("Initialized Environment (Grid)")
plt.imshow(grid, cmap='coolwarm')
plt.show()

rewards = reward_grid(grid, grid_size, reward_def, reward_goal, reward_obstacle)

plt.title("Initialized Rewards (Grid)")
plt.imshow(rewards, cmap='coolwarm')
plt.show()



# %%
def state_action(state, action, grid_size):
    x, y = state
    if (action == "up" and x > 0):
        return (x-1, y)
    elif (action == "down" and x < grid_size - 1):
        return (x+1, y)
    elif (action == "right" and y < grid_size - 1):
        return (x, y+1)
    elif (action == "left" and y > 0):
        return (x, y-1)
    return state

# %%
def value_iteration(rewards, gamma, threshold, goal, actions, grid_size):
    values = np.zeros_like(rewards, dtype=float)
    policy = np.empty_like(rewards, dtype=float)

    while (True):
        delta = 0
        for x in range(grid_size):
            for y in range(grid_size):
                if ((x, y) == goal or grid[x, y] == -1):
                    continue

                action_values = []
                for action in actions:
                    next_state = state_action((x, y), action, grid_size)
                    nx, ny = next_state
                    action_values.append(rewards[nx, ny] + gamma * values[nx, ny])
                
                best_action = actions[np.argmax(action_values)]
                best_value = max(action_values)
                delta = max(delta, abs(best_value - values[x, y]))
                # print(policy)
                print(next_state)

                values[x, y] = best_value
                policy[x, y] = best_action

        if delta < threshold:
            break

    return values, policy                

values_vi, policy_vi = value_iteration(rewards, gamma, threshold, goal, actions, grid_size)

# %%
def q_learning(rewards, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_values = np.zeros((grid_size, grid_size, len(actions)))

    for _ in range(episodes):
        state = start
        while state != goal:
            x, y = state
            
            # Choose action with epsilon-greedy policy
            if random.random() < epsilon:
                action_index = random.randint(0, len(actions) - 1)
            else:
                action_index = np.argmax(q_values[x, y])

            action = actions[action_index]
            next_state = state_action(state, action)
            nx, ny = next_state

            # Compute the reward and update Q-values
            reward = rewards[nx, ny]
            td_target = reward + gamma * np.max(q_values[nx, ny])
            q_values[x, y, action_index] += alpha * (td_target - q_values[x, y, action_index])

            state = next_state  # Move to next state
    
    # Extract policy from Q-values
    policy_q = np.empty_like(rewards, dtype=object)
    for x in range(grid_size):
        for y in range(grid_size):
            best_action_index = np.argmax(q_values[x, y])
            policy_q[x, y] = actions[best_action_index]
    
    return q_values, policy_q

# %%
q_values_ql, policy_ql = q_learning(rewards)

# %%
def plot_policy(policy, title):
    plt.figure(figsize=(10, 10))
    policy_grid = np.zeros((grid_size, grid_size), dtype=str)
    for x in range(grid_size):
        for y in range(grid_size):
            if grid[x, y] == -1:
                policy_grid[x, y] = 'X'  # Obstacle
            elif grid[x, y] == 2:
                policy_grid[x, y] = 'S'  # Start
            elif grid[x, y] == 3:
                policy_grid[x, y] = 'G'  # Goal
            else:
                policy_grid[x, y] = policy[x, y] if policy[x, y] is not None else ' '
    
    plt.imshow(grid, cmap='coolwarm', origin='upper')
    for (j, i), label in np.ndenumerate(policy_grid):
        plt.text(i, j, label, ha='center', va='center', color="black")
    plt.title(title)
    plt.show()

# %%
# Plot policies
plot_policy(policy_vi, "Optimal Policy (Value Iteration)")
plot_policy(policy_ql, "Optimal Policy (Q-Learning)")


