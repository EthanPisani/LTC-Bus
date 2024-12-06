import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import time
from torch.optim.lr_scheduler import LambdaLR

import pandas as pd
import numpy as np
import torch

class Standardizer:
    def __init__(self, input_features, target_feature):
        self.input_features = input_features
        self.target_feature = target_feature
        self.mean = None
        self.std = None

    def fit(self, X_train, y_train):
        # Calculate the mean and std for both input features and target
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)
        self.target_mean = y_train.mean()
        self.target_std = y_train.std()

    def transform(self, X):
        # Standardize features
        X_scaled = (X - self.mean) / self.std
        return X_scaled

    def inverse_transform(self, y_scaled):
        # Revert the scaling of the target
        return y_scaled * self.target_std + self.target_mean

    def transform_input(self, X):
        return self.transform(X)

    def transform_target(self, y):
        return (y - self.target_mean) / self.target_std

# Preprocessing function with stop_id embedding
def preprocess_data(filepath):
    # Load CSV
    df = pd.read_csv(filepath)

    # Circular encoding for the day of the week
    DAYS_IN_WEEK = 7
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / DAYS_IN_WEEK)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / DAYS_IN_WEEK)

    # One-hot encode the route_id
    df_route_one_hot = pd.get_dummies(df['route_id'], prefix='route')
    df = pd.concat([df, df_route_one_hot], axis=1)

    # Scale scheduled_time
    df['scheduled_time'] = df['scheduled_time'] / 86400  # Convert to fraction of day
    # prev arrival time
    df['prev_arrival_time'] = df['actual_time'].shift(1)
    df['prev_arrival_time'] = df['prev_arrival_time'].fillna(0)
    df['prev_arrival_time'] = df['prev_arrival_time'] / 86400  # Convert to fraction of day

    # Define input features
    features = [col for col in df.columns if col.startswith('route_')] + ['stop_id', 'day_sin', 'day_cos', 'scheduled_time', 'prev_arrival_time']

    # Remove unused columns
    df = df[features + ['delay']].dropna()

    X = df[features]
    y = df['delay']  # Target variable (arrival time)

    
    # shift all delay values to positive
    # save min
    y_min = y.min()
    print("y min", y_min)

    y = y - y.min()
    # save max
    y_max = y.max()
    print("y max", y_max)

    # save mean
    y_mean = y.mean()
    print("y mean", y_mean)
    # save std
    y_std = y.std()
    print("y std", y_std)
    # Normalize target values

    y = (y - y.mean()) / y.std()  # Normalize target values
    print("y max", y.max())
    print("y min", y.min())
    return X, y


class ArrivalTimeModel64x32(nn.Module):
    def __init__(self, num_routes, num_stops, stop_emb_dim=48):
        super(ArrivalTimeModel64x32, self).__init__()
        self.stop_emb = nn.Embedding(num_stops, stop_emb_dim)
        self.fc1 = nn.Linear(num_routes + stop_emb_dim + 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, route_ids, stop_ids, day_features, scheduled_time, prev_arrival_time):
        stop_embedded = self.stop_emb(stop_ids).view(stop_ids.size(0), -1)
        # print shapes of all
        # print("route_ids", route_ids.shape)
        # print("stop_ids", stop_ids.shape)
        # print("day_features", day_features.shape)
        # print("scheduled_time", scheduled_time.shape)
        # print("prev_arrival_time", prev_arrival_time.shape)

        x = torch.cat([route_ids, stop_embedded, day_features, scheduled_time, prev_arrival_time], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ArrivalTimeModel128x64x32(nn.Module): # not worth
    def __init__(self, num_routes, num_stops, stop_emb_dim=128):
        super(ArrivalTimeModel128x64x32, self).__init__()
        print("values",num_stops, stop_emb_dim)
        self.stop_emb = nn.Embedding(num_stops, stop_emb_dim)
        self.fc1 = nn.Linear(num_routes + stop_emb_dim + 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, route_ids, stop_ids, day_features, scheduled_time, prev_arrival_time):
        stop_embedded = self.stop_emb(stop_ids).view(stop_ids.size(0), -1)
        x = torch.cat([route_ids, stop_embedded, day_features, scheduled_time, prev_arrival_time], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(filepath, output_dir, num_epochs=50000, learning_rate=0.001,save_every=10000):
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess data
    X, y = preprocess_data(filepath)



    # Convert to tensors
    route_ids = torch.tensor(X[[col for col in X.columns if col.startswith('route_')]].astype(np.float32).values, dtype=torch.bfloat16)
    stop_ids = torch.tensor(X[['stop_id']].values, dtype=torch.long)
    day_features = torch.tensor(X[['day_sin', 'day_cos']].values, dtype=torch.bfloat16)
    scheduled_time = torch.tensor(X[['scheduled_time']].values, dtype=torch.bfloat16)
    prev_arrival_time = torch.tensor(X[['prev_arrival_time']].values, dtype=torch.bfloat16)
    targets = torch.tensor(y.values, dtype=torch.bfloat16).view(-1, 1)



    # Create TensorDataset and split into train, validation, and test sets
    dataset = TensorDataset(route_ids, stop_ids, day_features, scheduled_time, prev_arrival_time, targets)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert dataset to tensors for direct processing
    route_ids_train, stop_ids_train, day_features_train, scheduled_time_train, prev_arrival_time_train, targets_train = train_dataset[:]

    # send to gpu
    route_ids_train = route_ids_train.to(device, dtype=torch.bfloat16)
    stop_ids_train = stop_ids_train.to(device, dtype=torch.long)
    day_features_train = day_features_train.to(device, dtype=torch.bfloat16)
    scheduled_time_train = scheduled_time_train.to(device, dtype=torch.bfloat16)
    prev_arrival_time_train = prev_arrival_time_train.to(device, dtype=torch.bfloat16)
    targets_train = targets_train.to(device, dtype=torch.bfloat16)


    route_ids_val, stop_ids_val, day_features_val, scheduled_time_val, prev_arrival_time_val, targets_val = val_dataset[:]
    # send to gpu
    route_ids_val = route_ids_val.to(device, dtype=torch.bfloat16)
    stop_ids_val = stop_ids_val.to(device, dtype=torch.long)
    day_features_val = day_features_val.to(device, dtype=torch.bfloat16)
    scheduled_time_val = scheduled_time_val.to(device, dtype=torch.bfloat16)
    prev_arrival_time_val = prev_arrival_time_val.to(device, dtype=torch.bfloat16)
    targets_val = targets_val.to(device, dtype=torch.bfloat16)


    route_ids_test, stop_ids_test, day_features_test, scheduled_time_test, prev_arrival_time_test, targets_test = test_dataset[:]

    route_ids_test = route_ids_test.to(device, dtype=torch.bfloat16)
    stop_ids_test = stop_ids_test.to(device, dtype=torch.long)
    day_features_test = day_features_test.to(device, dtype=torch.bfloat16)
    scheduled_time_test = scheduled_time_test.to(device, dtype=torch.bfloat16)
    prev_arrival_time_test = prev_arrival_time_test.to(device, dtype=torch.bfloat16)
    targets_test = targets_test.to(device, dtype=torch.bfloat16)



    # Model setup
    num_routes = route_ids.shape[1]
    num_stops = X['stop_id'].nunique() + 1
    model = ArrivalTimeModel64x32(num_routes=num_routes, num_stops=num_stops, stop_emb_dim=48)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).to(torch.bfloat16)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_model = model.state_dict()
    best_loss = float('inf')



    # gen train session filelog file name
    train_session_filelog = os.path.join(output_dir, f"train_session{time.time()}.log")
    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()

        # Forward pass (process entire training dataset)
        outputs = model(route_ids_train, stop_ids_train, day_features_train, scheduled_time_train, prev_arrival_time_train)
        loss = criterion(outputs, targets_train)

        # Backward pass and optimization
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Print training loss
        if epoch % 250 == 0:
            tqdm.write(f"Epoch {epoch}, Training Loss: {loss.item():.4f}")
            with open(train_session_filelog, 'a') as f:
                f.write(f"Epoch {epoch}, Training Loss: {loss.item():.4f}\n")
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model.state_dict()
                tqdm.write(f"Best model updated at epoch {epoch}")
                with open(train_session_filelog, 'a') as f:
                    f.write(f"Best model updated at epoch {epoch}\n")
                torch.save(best_model, os.path.join(output_dir, f"best_{model._get_name()}_bf16.pt"))
        # Validation
        if epoch % 1000 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(route_ids_val, stop_ids_val, day_features_val, scheduled_time_val, prev_arrival_time_val)
                val_loss = criterion(val_outputs, targets_val)
                tqdm.write(f"Epoch {epoch}, Validation Loss: {val_loss.item():.4f}")
                with open(train_session_filelog, 'a') as f:
                    f.write(f"Epoch {epoch}, Validation Loss: {val_loss.item():.4f}\n")

        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"{model._get_name()}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, "model_bf16_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to: {final_model_path}")


    return model, test_dataset



if __name__ == "__main__":
    filepath = "./extracted/combined.csv"
    output_dir = "./checkpoints"
    trained_model, test_loader = train_model(filepath, output_dir, num_epochs=90000, learning_rate=0.001, save_every=10000)
