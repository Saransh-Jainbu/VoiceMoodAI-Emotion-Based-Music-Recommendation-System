import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import pickle
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Load data
Emotions = pd.read_csv('emotion.csv')
Emotions = Emotions.fillna(0)

X = Emotions.iloc[:, :-1].values
Y = Emotions['Emotions'].values
print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')

# Encode labels
encoder = OneHotEncoder()
Y_encoded = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, random_state=42, test_size=0.2, shuffle=True)

# Reshape for 1D CNN (add channel dimension)
X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])  # (batch, channels, features)
X_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# Scale features
scaler = StandardScaler()
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Reshape back for CNN
X_train_scaled = x_train_scaled.reshape(x_train.shape[0], 1, x_train.shape[1])
X_test_scaled = x_test_scaled.reshape(x_test.shape[0], 1, x_test.shape[1])

print(f'Training data shape: {X_train_scaled.shape}')
print(f'Test data shape: {X_test_scaled.shape}')

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout3 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 75, 512)  # Corrected input size after conv layers
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout1(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout2(x)

        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = torch.relu(self.bn6(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model
model = EmotionCNN().to(device)
print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
best_accuracy = 0.0
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels_max = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == labels_max).sum().item()

    train_accuracy = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()

    val_accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {val_accuracy:.2f}%')

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_emotion_model.pth')
        patience_counter = 0
        print(f'Best model saved with accuracy: {best_accuracy:.2f}%')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Save final model
torch.save(model.state_dict(), 'emotion_model.pth')
print('Training completed!')

# Save scaler and encoder
with open('scaler2.pickle', 'wb') as f:
    pickle.dump(scaler, f)

with open('encoder2.pickle', 'wb') as f:
    pickle.dump(encoder, f)

print('Scaler and encoder saved!')