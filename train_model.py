import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet
from dataset import CircleDataset

IMG_SIZE = 128
TOTAL_SAMPLES = 1000

dataset = CircleDataset(size=IMG_SIZE, n_samples=TOTAL_SAMPLES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    model.train()
    epoch_loss = 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "trained_model.pth")
