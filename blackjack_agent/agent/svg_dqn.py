# svg_dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from .mlp import MLPQNetwork  # Import the MLPQNetwork class

class SVG_DQN:
    def __init__(self, input_size, hidden_sizes, output_size, gamma=0.99, lr=0.001, buffer_size=10000):
        """
        Initializes the SVG-DQN agent.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list): List of hidden layer sizes.
            output_size (int): Number of possible actions.
            gamma (float): Discount factor for future rewards.
            lr (float): Learning rate for the optimizer.
            buffer_size (int): Maximum size of the replay buffer.
        """
        # Main Q-network and target Q-network
        self.q_network = MLPQNetwork(input_size, hidden_sizes, output_size)
        self.target_network = MLPQNetwork(input_size, hidden_sizes, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set the target network to evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self, batch_size):
        """Samples a batch of experiences from the replay buffer."""
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def select_action(self, state, epsilon):
        """
        Selects an action using the epsilon-greedy policy.

        Args:
            state (torch.Tensor): Current game state.
            epsilon (float): Probability of selecting a random action.

        Returns:
            int: Selected action.
        """
        if random.random() < epsilon:
            return random.randint(0, self.q_network.model[-1].out_features - 1)  # Random action
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                return q_values.argmax().item()  # Greedy action

    def update_q_network(self, batch_size):
        """Performs a training step on the Q-network using a batch from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return  # Wait until we have enough samples in the replay buffer

        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)

        # Get Q-values for the current states and the actions taken
        q_values = self.q_network(states)  # Shape: [batch_size, num_actions]
       

        q_values = q_values.gather(1, actions).squeeze(1)  # Gather Q-values for each taken action

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss and perform backpropagation
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Updates the target network to match the main Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
