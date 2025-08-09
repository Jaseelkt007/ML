import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

###### MONTAINCAR DEMONSTRATION ######
# This code demonstrates how to use Q learning with MountainCar environment.

# Discretize the continuous state space 
"""Idea : Sate aggregation
1. Take continuous position, velocity.
2. Map them to integer bin indices(i_pos, i_vel)
3. Use that tuple as the “state ID” for a table.
"""

def create_bins(num_bins, low, high):
    """Create evenly space bins for a continuous space."""
    return np.linspace(low, high, num_bins+1)[1:-1]

def discretize_state(state, pos_bins, vel_bins):
    """ Get the bin indices for the position and velocity."""
    pos, vel = state
    i_pos = np.digitize(pos, pos_bins)
    v_pos = np.digitize(vel, vel_bins)
    return (i_pos, v_pos)

def e_greedy_action(Q_value, state, epsilon, env):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)  # Explore
    else:
        i_pos, i_vel = state
        return np.argmax(Q_value[i_pos, i_vel])


def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print(f"action take: {action}")
        state, reward, terminated, truncated, info = env.step(action)
        print(f"state: {state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
        done = terminated or truncated
        if done:
            print("Episode finished!")
            break

def train(env, num_episodes = 1000, alpha =0.1, gamma = 0.99, epsilon = 0.1):
    """Train loops: as follows :
    1.Initialize Q-table.
    2.Loop over episodes:
       1. Reset env.
       2. Discretize initial state.
       3. Loop steps:
          1. Choose action using epsilon-greedy policy - policy improvement step
          2. Take action, observe next state and reward.
          3. Discretize next state.
          4. Update Q-value using for eg: Q learning update rule.
          5. Move to next state.
    3. Track total rewards for ploting"""
    pos_bins = create_bins(20,env.observation_space.low[0], env.observation_space.high[0])[1:-1]
    vel_bins = create_bins(20,env.observation_space.low[1], env.observation_space.high[1])[1:-1]
    Q_value = np.zeros((len(pos_bins), len(vel_bins), env.action_space.n))
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        state = discretize_state(state, pos_bins, vel_bins)
        while True:
            action = e_greedy_action(Q_value, state, epsilon, env)
            next_state, rewards, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_state, pos_bins, vel_bins)
            # Q-learning update rule
            Q_value[state[0], state[1], action] += alpha * (rewards + gamma * np.max(Q_value[next_state[0], next_state[1]]) - Q_value[state[0],state[1], action])
            state = next_state
            total_rewards.append(rewards)
            if terminated or truncated:
                print(f"Episode {episode + 1} finished with total reward: {sum(total_rewards)}")
                break
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.show()

def main():
    env = gym.make('MountainCar-v0', render_mode="human")
    state, info = env.reset()
    train(env, num_episodes=10, alpha=0.1, gamma=0.99, epsilon=0.1)
    # random_episode(env)
    env.close()


if __name__ == "__main__":
    main()

# General notes:
"""
* Tabular Q-learning means: one Q-value for every discrete state–action pair.
* State aggregation (binning) is still a table — just a smaller one because we grouped continuous states into bins.
* This is not function approximation — it’s still lookup.
* Typical tabular Q-learning for MountainCar with 20×20 bins might need 10k–20k episodes to learn a decent policy.

"""