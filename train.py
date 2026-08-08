import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import ANN


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# DATA TRANSFORMATION
# ==========================================

transform = transforms.ToTensor()


# ==========================================
# LOAD DATASET
# ==========================================

dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

train_set = int(0.8 * len(dataset))
val_set = len(dataset) - train_set

train_dataset, val_dataset = random_split(
    dataset,
    [train_set, val_set]
)


# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ==========================================
# MODEL
# ==========================================

model = ANN().to(device)

print(model)


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = torch.nn.CrossEntropyLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ==========================================
# LEARNING RATE SCHEDULER
# ==========================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=3
)


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Add batch loss
        total_loss += loss.item()

    return total_loss / len(loader)


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predicted class
            predictions = outputs.argmax(dim=1)

            # Count correct predictions
            correct += (
                predictions == labels
            ).sum().item()

            # Count total samples
            total += labels.size(0)

    loss = total_loss / len(loader)

    accuracy = 100 * correct / total

    return loss, accuracy


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

best_val_loss = float("inf")

patience = 5

counter = 0

epochs = 50


# Create checkpoint directory
os.makedirs(
    "checkpoints",
    exist_ok=True
)


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(epochs):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion,
        device
    )

    # Update learning rate
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.2f}%"
    )

    # Check if model improved
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        print("Best model saved!")

    else:

        counter += 1

    # Early stopping
    if counter >= patience:

        print("Early stopping!")

        break