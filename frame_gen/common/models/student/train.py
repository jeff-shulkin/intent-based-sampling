import torch
import torchaudio
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from time import time

from test import test_model, test_single_sample_inference

# Ensure GPU/CPU compatibility
DTYPE = torch.float
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)

# Load datasets
dopplerDataset = DopplerSpectrogramDataset()

# Split dataset: 70% train, 15% val, 15% test
train_size = int(0.7 * len(dopplerDataset))
val_size = int(0.15 * len(dopplerDataset))
test_size = len(dopplerDataset) - train_size - val_size
train_set, val_set, test_set = random_split(
    dopplerDataset, [train_size, val_size, test_size]
)

# Create DataLoaders
batch_size = 8
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize model, loss function, and optimizer
learning_rate = 1e-3
model = DopplerCNN().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Training function
def train_model():
    train_loss_history = []
    train_accuracy_history = []
    for epoch in range(10):
        start_time = time()

        # =====================================================================
        #  TRAIN
        # =====================================================================

        model.train() # Set model to train
        epoch_loss_history = []
        num_correct = 0
        num_total = 0

        for inputs, true_labels in train_dl:
            inputs, true_labels = inputs.to(DEVICE), true_labels.to(DEVICE)
            true_labels = true_labels.squeeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, true_labels)
            epoch_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            _, predicted_labels = torch.max(outputs, 1)
            num_total += true_labels.size(0)
            num_correct += (predicted_labels == true_labels).sum().item()

        train_loss_history.append(sum(epoch_loss_history) / len(epoch_loss_history))
        train_accuracy_history.append(100 * num_correct / num_total)

        torch.cuda.empty_cache()

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss_history[-1]}")
        print(f"Training accuracy: {train_accuracy_history[-1]}")
        validate_model(model)


# Validation function
def validate_model(model):

    # Variables to assess performance
    epoch_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for inputs, true_labels in val_dl:
            inputs, true_labels = inputs.to(DEVICE), true_labels.to(DEVICE)
            true_labels = true_labels.squeeze(1)
            outputs = model(inputs)
            loss = loss_function(outputs, true_labels)
            epoch_loss_history.append(loss.item())
            _, predicted_labels = torch.max(outputs, 1)
            num_total += true_labels.size(0)
            num_correct += (predicted_labels == true_labels).sum().item()

    validation_loss_history.append(sum(epoch_loss_history) / len(epoch_loss_history))
    validation_accuracy_history.append(100 * num_correct / num_total)
    print(f"Validation loss: {validation_loss_history[-1]}")
    print(f"Validation accuracy: {validation_accuracy_history[-1]} %")

# Train the model
train_model()


# Test model
test_model(model, test_dl)

# Test single inference time on CPU
test_single_sample_inference(model, test_dl)
torch.save(model.state_dict(), "model.pth")