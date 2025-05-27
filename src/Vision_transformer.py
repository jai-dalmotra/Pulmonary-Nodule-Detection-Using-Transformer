# Imports
# -------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from tqdm import tqdm
import random
import math

# -------------------------------------------------
# Config
# -------------------------------------------------
CT_SCANS_DIR = '/kaggle/input/luna-lung-cancer-dataset/seg-lungs-LUNA16/seg-lungs-LUNA16'
ANNOTATIONS_CSV = '/kaggle/input/luna-lung-cancer-dataset/annotations.csv'
CANDIDATES_CSV = '/kaggle/input/luna-lung-cancer-dataset/candidates.csv'
PATCH_SIZE = 32
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(itkimage)
    origin = np.array(itkimage.GetOrigin())[::-1]
    spacing = np.array(itkimage.GetSpacing())[::-1]
    return img, origin, spacing

def world_to_voxel(world_coord, origin, spacing):
    stretched = np.abs(world_coord - origin)
    voxel_coord = stretched / spacing
    return voxel_coord

def extract_patch(img, center, size):
    center = [int(c) for c in center]
    size = [int(s) for s in (size, size, size)]
    start = [max(c - s//2, 0) for c, s in zip(center, size)]
    end = [start[i] + size[i] for i in range(3)]
    slices = tuple(slice(start[i], end[i]) for i in range(3))
    patch = img[slices]
    if patch.shape != (size[0], size[1], size[2]):
        pad_width = [(0, max(0, size[i] - patch.shape[i])) for i in range(3)]
        patch = np.pad(patch, pad_width, mode='constant', constant_values=-1000)
    return patch

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class LunaDataset(Dataset):
    def _init_(self, candidates_file, ct_dir, transform=None):
        self.df = pd.read_csv(candidates_file)
        self.ct_dir = ct_dir
        self.transform = transform
        self.cache = {}

    def _len_(self):
        return len(self.df)

    def _getitem_(self, idx):
        row = self.df.iloc[idx]
        seriesuid = row['seriesuid']
        world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
        label = row['class']

        if seriesuid not in self.cache:
            img, origin, spacing = load_itk(os.path.join(self.ct_dir, seriesuid + '.mhd'))
            self.cache[seriesuid] = (img, origin, spacing)
        else:
            img, origin, spacing = self.cache[seriesuid]

        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        patch = extract_patch(img, voxel_coord, PATCH_SIZE)
        patch = np.clip(patch, -1000, 400)
        patch = (patch + 1000) / 1400

        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return patch, label

# -------------------------------------------------
# Model
# -------------------------------------------------
class SimpleViT3D(nn.Module):
    def _init_(self):
        super(SimpleViT3D, self)._init_()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x.squeeze(1)

# -------------------------------------------------
# Training and Evaluation
# -------------------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    preds = []
    truths = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            truths.append(labels.cpu().numpy())
            preds_binary = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds_binary == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return running_loss / len(loader), accuracy, np.concatenate(preds), np.concatenate(truths)

# -------------------------------------------------
# Main
# -------------------------------------------------
train_dataset = LunaDataset(CANDIDATES_CSV, CT_SCANS_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = SimpleViT3D().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = eval_epoch(model, train_loader, criterion)

    train_losses.append(train_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Acc {val_acc:.4f}")

# -------------------------------------------------
# Plotting
# -------------------------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.show()
