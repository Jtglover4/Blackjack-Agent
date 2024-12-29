import torch
import torch.nn as nn


class MLPQNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes,output_size):
        """

        Initializes the MLP Q-network model and paremeters

        Args:
            input_size (int): The number of input features (e.g., player's hand, dealer's visible card, usable ace).
            hidden_sizes (list): List of integers for the number of neurons in each hidden layer (e.g., [128, 64]).
            output_size (int): The number of possible actions (e.g., 2 for "hit" and "stand").
        """


        #ensuring model inherits methods of nn.module
        super(MLPQNetwork,self).__init__()

        #defining each layer of our MLP

        # used sequential because each layers output directly feeds into the next layer
        self.model = nn.Sequential(
            #the first fully connected dense layer, with input size input features, and hidden_sizes[0] output layer so passing the 128
            nn.Linear(input_size,hidden_sizes[0]),
            #ReLU activation function
            nn.ReLU(),
            #takes in 128 neurons, passes out 64
            nn.Linear(hidden_sizes[0],hidden_sizes[1]),
            #ReLU activation function
            nn.ReLU(),
            #taking 64 neurons and outputing hit or stand
            nn.Linear(hidden_sizes[1],output_size) #Output layer: Q-values for each action
        )


    def forward(self,x):
        """
            Forward pass of the network.
        
            Args:
                x (torch.Tensor): Input tensor representing the current state.

            Returns:
        torch.Tensor: Q-values for each action."""
        return self.model(x)
    
    def save(self,file_path):
        """Saves the model parameters to the specified file path."""
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    
    def load(self, file_path):
        """Loads the model parameters from the specified file path."""
        self.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")
    




        


        