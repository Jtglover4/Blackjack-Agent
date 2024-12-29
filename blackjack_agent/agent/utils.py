# utils.py

import torch
import os
import csv


def save_model(model, file_path):
    """Saves the model to the specified file path."""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path):
    """Loads the model from the specified file path."""
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")

def log_metrics(episode, reward, epsilon, loss):
    """Logs the metrics to the console or file."""
    print(f"Episode: {episode}, Reward: {reward}, Epsilon: {epsilon}, Loss: {loss}")

def log_metrics_to_csv(episode, reward, epsilon, loss, file_path="training_metrics.csv"):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as csv_file:
        fieldnames = ['Episode', 'Reward', 'Epsilon', 'Loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Episode': episode, 'Reward': reward, 'Epsilon': epsilon, 'Loss': loss})

    print(f"Episode: {episode}, Reward: {reward}, Epsilon: {epsilon:.3f}, Loss: {loss:.6f}")
