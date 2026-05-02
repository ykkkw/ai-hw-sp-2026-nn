import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import MLP
from model import CNN
from model import Encoder


# model settings (change model name each time)
MODEL_NAME   = "encoder"
EPOCHS       = 15
BATCH_SIZE   = 64
LEARN_RATE   = 0.001
VAL_SPLIT    = 0.2
SAVE_PATH    = f"{MODEL_NAME}_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# load model
if MODEL_NAME == "mlp":
    model = MLP()
elif MODEL_NAME == "cnn":
    model = CNN()
elif MODEL_NAME == "encoder":
    model = Encoder()

# load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# download full training set (60000 images)
full_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)

# split into train + validation; 60000 × 0.8 = 48000 train; 60000 × 0.2 = 12000 validation
val_size   = int(len(full_train) * VAL_SPLIT)
train_size = len(full_train) - val_size

train_dataset, val_dataset = random_split(
    full_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train count: {len(train_dataset)}")
print(f"Validation count: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

model = model.to(device)
print(f"Model: {MODEL_NAME} | Params: {sum(p.numel() for p in model.parameters()):,}")

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


# training
best_val_accuracy = 0.0
best_val_accuracy_epoch = 0

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ──────────────────────────────
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_images  = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_images  += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = total_correct / total_images

    # validation data to evaluate
    model.eval()
    val_correct = 0
    val_images  = 0

    with torch.no_grad():
        for images, labels in val_loader: 
            images = images.to(device)
            labels = labels.to(device)
            preds  = model(images).argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_images  += labels.size(0)

    val_acc = val_correct / val_images

    print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}%") 


    # save based on VALIDATION accuracy, not test accuracy
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_val_accuracy_epoch = epoch
        torch.save(model.state_dict(), SAVE_PATH)
    
print(f"Finish training! Best val accuracy: {best_val_accuracy*100:.1f}% "
      f"at epoch {best_val_accuracy_epoch}")