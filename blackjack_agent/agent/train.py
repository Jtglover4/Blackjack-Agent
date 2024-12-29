# train.py
import os
import torch
import numpy as np
import time
from agent.mlp import MLPQNetwork
from agent.svg_dqn import SVG_DQN
from agent.epsilon_greedy import EpsilonGreedyPolicy
from agent.replay_buffer import ReplayBuffer
from agent.utils import save_model, load_model, log_metrics, log_metrics_to_csv  # assuming utils.py has these functions
import gym  # Assuming Blackjack-v1 environment from OpenAI Gym

# Hyperparameters
INPUT_SIZE = 3                  # Example: player's hand value, dealer's card, usable ace
HIDDEN_SIZES = [128, 64]        # Hidden layer sizes for MLP
OUTPUT_SIZE = 2                 # Actions: hit or stand
GAMMA = 0.99                    # Discount factor
LR = 0.0001                     # Learning rate
BUFFER_SIZE = 10000             # Replay buffer size
BATCH_SIZE = 64                 # Batch size for training
EPSILON_START = 1.0             # Initial exploration rate
EPSILON_END = 0.1               # Minimum exploration rate
EPSILON_DECAY = 0.995           # Decay rate for epsilon
TARGET_UPDATE_FREQ = 100        # Frequency of updating target network
NUM_EPISODES = 10000             # Total training episodes
SAVE_FREQ = 500                 # Save model every X episodes

# Initialize environment
env = gym.make("Blackjack-v1")

# Initialize agent components
q_network = MLPQNetwork(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
svg_dqn_agent = SVG_DQN(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, gamma=GAMMA, lr=LR, buffer_size=BUFFER_SIZE)
epsilon_policy = EpsilonGreedyPolicy(start_epsilon=EPSILON_START, end_epsilon=EPSILON_END, decay_rate=EPSILON_DECAY)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# Track metrics
all_rewards = []
losses = []
epsilon_values = []

# Training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0
    loss = None

    while not done:
        # Select action using epsilon-greedy policy
        epsilon = epsilon_policy.epsilon
        action = svg_dqn_agent.select_action(state, epsilon)

        # Take action in the environment
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

        # Store experience in replay buffer
        svg_dqn_agent.store_experience(state, action, reward, next_state, done)
        state = next_state

        # Update Q-network if there are enough experiences in the replay buffer
        if len(svg_dqn_agent.replay_buffer) >= BATCH_SIZE:
            loss = svg_dqn_agent.update_q_network(BATCH_SIZE)
            losses.append(loss)

    # Decay epsilon after each episode
    epsilon_policy.decay_epsilon()
    epsilon_values.append(epsilon_policy.epsilon)
    all_rewards.append(total_reward)

    # Update target network periodically
    if episode % TARGET_UPDATE_FREQ == 0:
        svg_dqn_agent.update_target_network()

    # Logging and checkpoints
    if episode % SAVE_FREQ == 0:
        save_model(svg_dqn_agent.q_network, f"/Users/sidhanthkapila/Desktop/blackjack_agent_566/data/checkpoints/final_model.pth")
        last_loss = loss if loss is not None else (losses[-1] if losses else 0)
        log_metrics_to_csv(episode, total_reward, epsilon_policy.epsilon, last_loss)
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon_policy.epsilon}, Loss: {last_loss}")

# Save final model and log final metrics
save_model(svg_dqn_agent.q_network, "/Users/sidhanthkapila/Desktop/blackjack_agent_566/data/checkpoints/final_model.pth")
last_loss = losses[-1] if losses else 0
log_metrics_to_csv(NUM_EPISODES, total_reward, epsilon_policy.epsilon, last_loss)
print(f"Training completed. Final model saved. Final loss: {last_loss}")


