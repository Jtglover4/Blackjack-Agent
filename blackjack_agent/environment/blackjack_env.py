import gymnasium as gym

class BlackjackEnvWrapper:
    def __init__(self, render_mode=None):
        """
        Initializes the Blackjack environment using Gymnasium's built-in environment.

        Args:
            render_mode (str): Optional; "human" for displaying the game (if available), None otherwise.
        """
        self.env = gym.make("Blackjack-v1", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        """
        Resets the environment for a new game.

        Returns:
            tuple: The initial state and additional info (player's hand value, dealer's visible card, usable ace).
        """
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        """
        Takes an action in the environment.

        Args:
            action (int): The action to take (0 for stand, 1 for hit).

        Returns:
            tuple: (next_state, reward, done, truncated, info)
                - next_state (tuple): The next state after taking the action.
                - reward (float): The reward obtained after taking the action.
                - done (bool): True if the game has ended, False otherwise.
                - truncated (bool): Whether the game was truncated.
                - info (dict): Additional information about the game.
        """
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done, truncated, info

    def close(self):
        """
        Closes the environment, cleaning up resources.
        """
        self.env.close()

    def render(self):
        """
        Renders the environment, if rendering is enabled.
        """
        self.env.render()