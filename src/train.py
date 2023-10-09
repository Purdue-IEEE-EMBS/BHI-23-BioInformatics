import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Regression Model for Multi-Target Output
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.00005
epochs = 150
batch_size = 32

# Load preprocessed data
data_X = np.load('data_X.npy')
data_Y = np.load('data_Y.npy')

# Assuming data_X has 4096 features
assert data_X.shape[1] == 4096, "Input data does not have 4096 features"

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(data_X))
test_size = len(data_X) - train_size
train_dataset, test_dataset = random_split(TensorDataset(torch.tensor(data_X, dtype=torch.float32).to(device), torch.tensor(data_Y, dtype=torch.float32).to(device)), [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss and Optimizer
model = RegressionModel(input_size=data_X.shape[1], output_size=data_Y.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Variables to store performance metrics
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0  # Reset epoch train loss
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
    for batch_X, batch_Y in train_progress_bar:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()  # Accumulate batch loss
        train_progress_bar.set_postfix(loss=loss.item())  # Update progress bar with current batch loss
    
    train_losses.append(epoch_train_loss/len(train_loader))  # Store average train loss for the epoch
    
    # Validation
    model.eval()
    total_loss = 0
    val_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
    with torch.no_grad():
        for batch_X, batch_Y in val_progress_bar:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            total_loss += loss.item()
            val_progress_bar.set_postfix(val_loss=loss.item())  # Update progress bar with current batch loss
        
        avg_loss = total_loss / len(test_loader)
        val_losses.append(avg_loss)  # Store average validation loss for the epoch
        val_progress_bar.set_postfix(val_loss=avg_loss)  # Update progress bar with average validation loss


torch.save(model.state_dict(), 'regressionmodel2.pth')
print("Model training complete. Model saved as regressionmodel2.pth")

# Plotting
fig, ax = plt.subplots()
ax.plot(range(1, epochs+1), train_losses, label='Training Loss')
ax.plot(range(1, epochs+1), val_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig('loss_over_epoch.png')
print("Loss over epoch plot saved as loss_over_epoch.png")

# Note: Since this is a regression problem, accuracy isn't a typical metric. 
# You may want to compute and plot R-squared values or other relevant metrics for training and validation instead.
