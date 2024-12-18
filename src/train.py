import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ResnetsegmentationModel
from dataset import FloodDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
num_classes = 2
batch_size = 4
learning_rate = 0.001
epochs = 10

#initialize dataset and model
train_dataset = FloodDataset('data/train/images', 'data/train/masks')
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)

model = ResnetsegmentationModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.long().to(device)
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "flood_model.pth")