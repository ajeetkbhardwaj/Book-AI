# save as train_mnist_conv.py and run with: python train_mnist_conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import numpy as np

# -------------------------
# Reproducibility / device
# -------------------------
seed = 20190402
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Hyperparameters
# -------------------------
batch_size = 64
test_batch_size = 1000
lr = 0.01
momentum = 0.9
epochs = 3           # change to larger value for full training
log_batch_every = 100

# -------------------------
# Data: MNIST
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),               # gives [0,1] float tensor (C,H,W) with C=1
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# Model (single Conv layer -> Flatten -> Dense)
# -------------------------
class SimpleConvNet(nn.Module):
    def __init__(self, out_channels=32, kernel_size=5):
        super().__init__()
        # input channels = 1 (MNIST grayscale)
        padding = kernel_size // 2    # "same" padding
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, bias=True)
        self.act = nn.Tanh()          # book used Tanh in example
        self.flatten = nn.Flatten()
        # compute flatten size: out_channels * H * W ; MNIST H=W=28
        flat_size = out_channels * 28 * 28
        self.fc = nn.Linear(flat_size, 10)   # 10 classes

        # weight init (Glorot/Xavier)
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (N,1,28,28)
        x = self.conv(x)           # (N, out_channels, 28,28)
        x = self.act(x)
        x = self.flatten(x)        # (N, out_channels*28*28)
        x = self.fc(x)             # (N,10)
        return x

model = SimpleConvNet(out_channels=32, kernel_size=5).to(device)

# -------------------------
# Loss + Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()   # combines LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# -------------------------
# Training loop with batch logging
# -------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = 100.0 * correct / len(loader.dataset)
    return avg_loss, acc

train_batches = len(train_loader)
for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_batch_every == 0:
            # evaluate quickly on test set partial or full
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch} Batch {batch_idx}/{train_batches} "
                  f"TrainLoss={loss.item():.4f} TestLoss={test_loss:.4f} TestAcc={test_acc:.2f}%")

    t1 = time.time()
    epoch_test_loss, epoch_test_acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch} completed in {t1-t0:.1f}s. "
          f"TestLoss={epoch_test_loss:.4f} TestAcc={epoch_test_acc:.2f}%")

# -------------------------
# Final evaluation
# -------------------------
final_loss, final_acc = evaluate(model, test_loader, device)
print(f"Final Test Loss: {final_loss:.4f}  Final Test Acc: {final_acc:.2f}%")
