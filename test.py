import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ANN


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# LOAD MODEL
# ==========================================

model = ANN().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

print("Best model loaded successfully!")


# ==========================================
# TRANSFORMATION
# ==========================================

transform = transforms.ToTensor()


# ==========================================
# LOAD TEST DATASET
# ==========================================

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# TEST DATALOADER
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ==========================================
# TEST MODEL
# ==========================================

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


# ==========================================
# CALCULATE ACCURACY
# ==========================================

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")