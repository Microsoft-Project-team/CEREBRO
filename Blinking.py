import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# === CONFIG ===
train_dir =r"D:\Cerebro1\Blink\train"
test_dir = r"D:\Cerebro1\Blink\test"
num_classes = 2
num_epochs = 3
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === DATASETS & LOADERS ===
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# === MODEL ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# === LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# === TRAINING ===
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = 100 * correct / len(train_dataset)

    # === TESTING ===
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())

    test_acc = accuracy_score(test_labels, test_preds) * 100
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# === SAVE MODEL ===
torch.save(model.state_dict(), "eye_state_resnet18.pth")
print("✅ Model saved to 'eye_state_resnet18.pth'")
