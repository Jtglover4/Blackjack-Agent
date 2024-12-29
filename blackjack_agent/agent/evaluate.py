# evaluate.py
import os
import csv
import torch
import gym
from .svg_dqn import SVG_DQN
from .mlp import MLPQNetwork
from .utils import load_model
from .shapley_shubik import ShapleyShubikExplainer
import numpy as np

# Hyperparameters (matching those used in train.py)
INPUT_SIZE = 3           # Example: player's hand value, dealer's card, usable ace
HIDDEN_SIZES = [128, 64] # Hidden layer sizes for MLP
OUTPUT_SIZE = 2          # Actions: hit or stand
GAMMA = 0.99
LR = 0.001
BUFFER_SIZE = 10000

def evaluate_agent(env, agent, num_episodes=1000):
    """
    Evaluates the agent on the given environment without exploration (epsilon = 0.0).
    Returns the average reward across `num_episodes`.
    """
    total_rewards = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # Greedy action selection
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            state = next_state
            episode_reward += reward
        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    return average_reward

def evaluate_win_rate(env, agent, num_episodes=1000):
    """
    Evaluates the win rate of the agent by counting how many episodes result in positive rewards.
    """
    wins = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # Greedy action selection
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            state = next_state

            if done and reward > 0:
                wins += 1

    win_rate = wins / num_episodes
    return win_rate

def log_evaluation_metrics(average_reward, win_rate, file_path="evaluation_metrics.csv"):
    """
    Logs evaluation metrics to a CSV file for comparison.
    """
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as csv_file:
        fieldnames = ['AverageReward', 'WinRate']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'AverageReward': average_reward, 'WinRate': win_rate})

    print(f"Average Reward: {average_reward}, Win Rate: {win_rate * 100:.2f}%")

def evaluate_with_explainability(env, agent, explainer, feature_names, num_episodes=10):
    results = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state).reshape(1, -1)
        done = False
        total_reward = 0
        episode_explanations = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            explanations = explainer.explain_decision(state, action)
            contributions = np.array([explanations[feature] for feature in feature_names])
            abs_contributions = np.abs(contributions)
            total_contribution = np.sum(abs_contributions)
            if total_contribution != 0:
                percentages = (abs_contributions / total_contribution) * 100
            else:
                percentages = np.zeros_like(abs_contributions)
            normalized_explanations = dict(zip([f"{feature} (%)" for feature in feature_names], percentages))

            # Combine original and normalized explanations
            combined_explanations = {**explanations, **normalized_explanations}

            episode_explanations.append({
                "state": state.flatten().tolist(),
                "action": action,
                **combined_explanations
            })

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
            total_reward += reward
            state = next_state

        results.append({"Episode": episode, "TotalReward": total_reward, "Explanations": episode_explanations})
    return results

def log_detailed_results(results, feature_names, file_path="/Users/sidhanthkapila/Desktop/blackjack_agent_566/shap_with_explanations.csv"):
    flattened_results = []
    print(f"Saving detailed results to: {os.path.abspath(file_path)}")
    percentage_feature_names = [f"{feature} (%)" for feature in feature_names]
    all_feature_names = feature_names + percentage_feature_names

    for result in results:
        for explanation in result["Explanations"]:
            data_row = {
                "Episode": result["Episode"],
                "TotalReward": result["TotalReward"],
                "State": explanation["state"],
                "Action": explanation["action"],
            }
            # Add original Shapley values
            data_row.update({feature: explanation.get(feature, 0) for feature in feature_names})
            # Add normalized percentages
            data_row.update({feature: explanation.get(feature, 0) for feature in percentage_feature_names})
            flattened_results.append(data_row)

    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ["Episode", "TotalReward", "State", "Action"] + all_feature_names
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_results)

    print(f"Detailed results saved to {file_path}")


if __name__ == "__main__":
    # Initialize environment
    env = gym.make("Blackjack-v1")

    # Initialize Q-network and agent
    q_network = MLPQNetwork(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    svg_dqn_agent = SVG_DQN(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, gamma=GAMMA, lr=LR, buffer_size=BUFFER_SIZE)

    # Load the saved model
    load_model(svg_dqn_agent.q_network, "/Users/sidhanthkapila/Desktop/blackjack_agent_566/data/checkpoints/final_model.pth")

    feature_names = ["Player's Hand", "Dealer's Card", "Usable Ace"]

    explainer = ShapleyShubikExplainer(svg_dqn_agent.q_network, feature_names)


    # Evaluate the loaded model
    average_reward = evaluate_agent(env, svg_dqn_agent, num_episodes=1000)
    win_rate = evaluate_win_rate(env, svg_dqn_agent, num_episodes=1000)
    log_evaluation_metrics(average_reward, win_rate)

    # Evaluate with explainability (existing code)
    results = evaluate_with_explainability(env, svg_dqn_agent, explainer, feature_names, num_episodes=1000)

    # Add this line to log detailed results
    log_detailed_results(results, feature_names)
