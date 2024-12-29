# replay_buffer.py

import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initializes the Replay Buffer.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)  # Automatically removes the oldest experience when full

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken by the agent.
            reward (float): Reward received after taking the action.
            next_state (torch.Tensor): Next state after taking the action.
            done (bool): Flag indicating if the episode ended after this action.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample_experiences(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones), where each element is a torch.Tensor.
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack experiences into separate lists for each element
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert lists to torch tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the replay buffer.
        """
        return len(self.buffer)
