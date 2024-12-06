import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import pickle
# Preprocessing function to include stop_id embedding
# Preprocessing function without stop_id embedding
def preprocess_data(filepath):
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Circular encoding for the day of the week
    DAYS_IN_WEEK = 7
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / DAYS_IN_WEEK)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / DAYS_IN_WEEK)
    
    # Scale scheduled_time
    scaler = MinMaxScaler()
    df['scheduled_time_scaled'] = scaler.fit_transform(df[['scheduled_time']])
    
    # One-hot encode the route_id
    df_route_one_hot = pd.get_dummies(df['route_id'], prefix='route')
    df = pd.concat([df, df_route_one_hot], axis=1)

    # Define input features
    features = [col for col in df.columns if col.startswith('route_')] + ['stop_id', 'day_sin', 'day_cos', 'scheduled_time_scaled']

    # Remove unused columns
    df = df[features + ['actual_time']]
    X = df[features]
    y = df['actual_time']  # Target variable (arrival time)
    
    return X, y, scaler


class ArrivalTimeModel(nn.Module):
    def __init__(self, num_routes):
        super(ArrivalTimeModel, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(num_routes + 4, 64)  # num_routes + stop_id + day_sin/cos + scheduled_time
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, route_ids, stop_ids, day_features, scheduled_time):
        # Concatenate tensors
        x = torch.cat([route_ids, stop_ids, day_features, scheduled_time], dim=1)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(filepath, output_dir, num_epochs=50000, learning_rate=0.001, save_every=10000):
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess data
    X, y, scaler = preprocess_data(filepath)

    # Convert to tensors
    route_ids = torch.tensor(X[[col for col in X.columns if col.startswith('route_')]].astype(np.int32).values, dtype=torch.float32)
    stop_ids = torch.tensor(X[['stop_id']].values, dtype=torch.float32)
    day_features = torch.tensor(X[['day_sin', 'day_cos']].values, dtype=torch.float32)
    scheduled_time = torch.tensor(X[['scheduled_time_scaled']].values, dtype=torch.float32)
    targets = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    # Model setup
    num_routes = route_ids.shape[1]
    model = ArrivalTimeModel(num_routes=num_routes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move tensors to GPU
    route_ids, stop_ids, day_features, scheduled_time, targets = map(
        lambda x: x.to(device),
        (route_ids, stop_ids, day_features, scheduled_time, targets)
    )

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(route_ids, stop_ids, day_features, scheduled_time)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Display loss
        if epoch % 1000 == 0:
            tqdm.write(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to: {final_model_path}")
    return model, scaler


if __name__ == "__main__":
    filepath = "./extracted/combined.csv"
    output_dir = "./checkpoints"
    trained_model, scaler = train_model(filepath, output_dir, num_epochs=90000, learning_rate=0.001, save_every=10000)

# MAE 64, 32, 1
# Epoch 1000/90000, Loss: 30542.3145                                                                                                  
# Epoch 2000/90000, Loss: 28637.9844                                                                                                  
# Epoch 3000/90000, Loss: 26231.2012                                                                                                  
# Epoch 4000/90000, Loss: 14952.5410                                                                                                  
# Epoch 5000/90000, Loss: 12657.8262                                                                                                  
# Epoch 6000/90000, Loss: 9752.4463                                                                                                   
# Epoch 7000/90000, Loss: 5856.4287                                                                                                   
# Epoch 8000/90000, Loss: 1726.7356
# Epoch 9000/90000, Loss: 1157.7289
# Epoch 10000/90000, Loss: 919.0682                                                                                                   