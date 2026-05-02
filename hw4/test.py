import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP
from model import CNN
from model import Encoder

# test setting
MODEL_NAME = "encoder"
SAVE_PATH  = f"{MODEL_NAME}_best.pt"
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_NAME == "mlp":
    model = MLP()
elif MODEL_NAME == "cnn":
    model = CNN()
elif MODEL_NAME == "encoder":
    model = Encoder()

# load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# load the saved weights into the model
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

model = model.to(device)
model.eval()
print(f"Loaded {SAVE_PATH}")


# test loop
correct = 0
total   = 0
class_correct = [0] * 10
class_total   = [0] * 10

with torch.no_grad(): 
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions     = model(images)
        predicted_class = predictions.argmax(dim=1)

        # calculate overall accuracy
        correct += (predicted_class == labels).sum().item()
        total   += labels.size(0)

        # calculate accuracy by class
        for pred, label in zip(predicted_class, labels):
            class_correct[label] += (pred == label).item()
            class_total[label]   += 1


# print out results
print(f"\n{'─'*35}")
print(f"Model:  {MODEL_NAME.upper()}")
print(f"Overall Accuracy: {correct/total*100:.2f}%  ({correct}/{total})")
print(f"{'─'*35}")
print("Perclass accuracy:")
for digit in range(10):
    acc = class_correct[digit] / class_total[digit] * 100
    bar = "█" * int(acc / 5)
    print(f"  Class {digit}: {acc:5.1f}%  {bar}")
print(f"{'─'*35}")