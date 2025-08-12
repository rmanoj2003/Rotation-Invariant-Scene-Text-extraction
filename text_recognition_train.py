import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from timm import create_model
from models.vision_transformer import *
from data.recognition_dataloader import *

# Hyperparameters
batch_size = 192
epochs = 300
learning_rate = 1.0
img_size = 224
patch_size = 16
embed_dim = 192  # Tiny
num_heads = 3
depth = 12
num_classes = 96  # e.g., alphanumeric + special tokens
max_length = 25  # Max text length (excl. [GO], [s])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize
model = TextRecognition().to(device)
train_dataset = ()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 as padding
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # [batch, max_length, num_classes]
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "text_recognition.pth")