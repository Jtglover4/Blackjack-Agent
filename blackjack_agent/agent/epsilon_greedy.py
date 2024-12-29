# epsilon_greedy.py

import random

class EpsilonGreedyPolicy:
    def __init__(self, start_epsilon=1.0, end_epsilon=0.1, decay_rate=0.995):
        """
        Initializes the epsilon-greedy policy with parameters for epsilon decay.

        Args:
            start_epsilon (float): Initial value of epsilon (for maximum exploration).
            end_epsilon (float): Minimum value of epsilon (for minimal exploration).
            decay_rate (float): Decay rate for epsilon over time.
        """
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_rate = decay_rate

    def select_action(self, q_values):
        """
        Selects an action using the epsilon-greedy strategy.

        Args:
            q_values (torch.Tensor or list): Q-values predicted by the agent for each action.

        Returns:
            int: The selected action (index of the action).
        """
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, len(q_values) - 1)
        else:
            # Exploit: choose the action with the highest Q-value
            return q_values.argmax().item()

    def decay_epsilon(self):
        """
        Decays the epsilon value to balance exploration and exploitation.
        """
        self.epsilon = max(self.end_epsilon, self.epsilon * self.decay_rate)

    def reset(self):
        """
        Resets epsilon to the starting value, useful for training or testing restarts.
        """
        self.epsilon = self.start_epsilon
