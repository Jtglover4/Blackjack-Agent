import itertools
import numpy as np
import torch

class ShapleyShubikExplainer:
    def __init__(self, q_network, feature_names):
        """
        Initializes the Shapley-Shubik explainer.

        Args:
            q_network (MLPQNetwork): The trained Q-network to explain.
            feature_names (list): List of feature names corresponding to the input features.
        """
        self.q_network = q_network
        self.feature_names = feature_names

    def compute_marginal_contribution(self, state, action_index):
        """
        Computes the Shapley-Shubik value for each feature.

        Args:
            state (np.ndarray): Input state for which to compute contributions.
            action_index (int): The action being explained.

        Returns:
            dict: Marginal contributions for each feature.
        """
        num_features = state.shape[1]
        feature_indices = list(range(num_features))
        permutations = list(itertools.permutations(feature_indices))
        
        contributions = np.zeros(num_features)

        for perm in permutations:
            coalition_value = 0
            for i, feature_idx in enumerate(perm):
                # Mask features not in the coalition
                mask = np.zeros_like(state)
                mask[:, perm[:i + 1]] = state[:, perm[:i + 1]]

                # Compute Q-value for coalition
                coalition_tensor = torch.tensor(mask, dtype=torch.float32)
                with torch.no_grad():
                    q_values = self.q_network(coalition_tensor)
                
                marginal_value = q_values[0, action_index].item() - coalition_value
                coalition_value = q_values[0, action_index].item()

                # Add marginal contribution
                contributions[feature_idx] += marginal_value

        # Average contributions over all permutations
        contributions /= len(permutations)

        # Return contributions as a dictionary
        return dict(zip(self.feature_names, contributions))

    def explain_decision(self, state, action):
        """
        Explains the decision for a given state and action.

        Args:
            state (np.ndarray): Input state for which to explain the decision.
            action (int): The chosen action to explain.

        Returns:
            dict: Shapley-Shubik contributions for each feature.
        """
        state = state.reshape(1, -1)  # Ensure correct shape
        return self.compute_marginal_contribution(state, action)