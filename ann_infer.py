import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

# Preprocessing function with stop_id embedding
def preprocess_input_data(input_data):
    
    # print unqiue values of route_id, stop_id
    # print("unique values of route_id",input_data['route_id'].unique())
    # print("unique values of stop_id",input_data['stop_id'].unique().shape)
    # shuffling the data
    # calculate man and std of delay of whole
    print("mean of delay",input_data['delay'].mean())
    print("std of delay",input_data['delay'].std())
    mean_delay = input_data['delay'].mean()
    std_delay = input_data['delay'].std()
    input_data = input_data.sample(frac=0.1).reset_index(drop=True)
    # Assume input_data is a DataFrame with the same structure as the training data
    DAYS_IN_WEEK = 7
    input_data['day_sin'] = np.sin(2 * np.pi * input_data['day'] / DAYS_IN_WEEK)
    input_data['day_cos'] = np.cos(2 * np.pi * input_data['day'] / DAYS_IN_WEEK)

    # One-hot encode the route_id
    df_route_one_hot = pd.get_dummies(input_data['route_id'], prefix='route')
    input_data = pd.concat([input_data, df_route_one_hot], axis=1)

    # Scale scheduled_time
    input_data['scheduled_time'] = input_data['scheduled_time'] / 86400  # Convert to fraction of day

    # Define input features
    features = [col for col in input_data.columns if col.startswith('route_')] + ['stop_id', 'day_sin', 'day_cos', 'scheduled_time']
    
    input_data = input_data[features + ['delay']].dropna()

    route_ids = input_data[[col for col in input_data.columns if col.startswith('route_')]].values.astype(np.float32)
    stop_ids = input_data[['stop_id']].values.astype(np.int64)

    day_features = input_data[['day_sin', 'day_cos']].values.astype(np.float32)
    scheduled_time = input_data[['scheduled_time']].values.astype(np.float32)
    actual_delays = input_data['delay'].values.astype(np.float32)

    return torch.tensor(route_ids), torch.tensor(stop_ids), torch.tensor(day_features), torch.tensor(scheduled_time), torch.tensor(actual_delays), mean_delay, std_delay

class ArrivalTimeModel128x64x32(nn.Module):
    def __init__(self, num_routes, num_stops, stop_emb_dim=16):
        super(ArrivalTimeModel128x64x32, self).__init__()
        self.stop_emb = nn.Embedding(num_stops, stop_emb_dim)
        self.fc1 = nn.Linear(num_routes + stop_emb_dim + 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, route_ids, stop_ids, day_features, scheduled_time):
        stop_embedded = self.stop_emb(stop_ids).view(stop_ids.size(0), -1)
        x = torch.cat([route_ids, stop_embedded, day_features, scheduled_time], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class ArrivalTimeModel64x32(nn.Module):
    def __init__(self, num_routes, num_stops, stop_emb_dim=16):
        super(ArrivalTimeModel64x32, self).__init__()
        self.stop_emb = nn.Embedding(num_stops, stop_emb_dim)
        self.fc1 = nn.Linear(num_routes + stop_emb_dim + 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, route_ids, stop_ids, day_features, scheduled_time):
        stop_embedded = self.stop_emb(stop_ids).view(stop_ids.size(0), -1)
        x = torch.cat([route_ids, stop_embedded, day_features, scheduled_time], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_model(model_path, num_routes, num_stops, device):
    # Load the trained model
    model = ArrivalTimeModel64x32(num_routes=num_routes, num_stops=num_stops, stop_emb_dim=48)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def infer(model, input_data, device):
    # Get input tensors
    route_ids, stop_ids, day_features, scheduled_time, actual_delays, mean, std = preprocess_input_data(input_data)
    
    # Move tensors to the correct device
    route_ids, stop_ids, day_features, scheduled_time, actual_delays = route_ids.to(device), stop_ids.to(device), day_features.to(device), scheduled_time.to(device), actual_delays.to(device)
    
    # Inference
    with torch.no_grad():
        predicted_delay = model(route_ids, stop_ids, day_features, scheduled_time)
    unnormalized_delay = predicted_delay.cpu().numpy() * std + mean
    return unnormalized_delay, actual_delays.cpu().numpy()

def main():
    # Path to the trained model
    # model_path = "./checkpoints/best_ArrivalTimeModel128x64x32_bf16.pt"
    model_path = "./checkpoints/best_ArrivalTimeModel64x32_bf16_1.pt"

    # Load a sample of test data (this can be from any file or user input)
    input_data_path = "./extracted/combined.csv"
    input_data = pd.read_csv(input_data_path)

    # Device setup (use GPU if available, otherwise CPU)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # route_ids, stop_ids, day_features, scheduled_time, actual_delays = preprocess_input_data(input_data)
    # Define the number of routes and stops (these should match the training dataset)
    num_routes = 38  # Update this based on your training data
    num_stops = 1952  # Update this based on your training data

    # Load the trained model
    model = load_model(model_path, num_routes, num_stops, device)

    # Make predictions
    predicted_delays, actual_delays = infer(model, input_data, device)

    

    # Print the results (actual vs predicted delays)
    for idx in range(50):
        print(f"Sample {idx + 1} - Actual Delay: {actual_delays[idx]:.2f} minutes, Predicted Delay: {predicted_delays[idx][0]:.2f} minutes")

    # make a plot of 1000 samples time vs delay (predicted and actual)
    import matplotlib.pyplot as plt
    plt.plot(actual_delays[:1000], label='Actual Delay')
    plt.plot(predicted_delays[:1000], label='Predicted Delay')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Delay')
    plt.savefig('delay_plot.png')

if __name__ == "__main__":
    main()
