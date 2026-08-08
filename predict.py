import torch
from torchvision import transforms
from PIL import Image

from model import ANN


# =========================================================
# 1. Configuration
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_model.pth"

IMAGE_PATH = "digit.png"


# =========================================================
# 2. Load Model
# =========================================================

model = ANN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# =========================================================
# 3. Image Transformation
# =========================================================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# =========================================================
# 4. Load Image
# =========================================================

image = Image.open(IMAGE_PATH)

image = transform(image)


# =========================================================
# 5. Add Batch Dimension
# =========================================================

image = image.unsqueeze(0)

image = image.to(DEVICE)


# =========================================================
# 6. Prediction
# =========================================================

with torch.no_grad():

    output = model(image)

    prediction = output.argmax(dim=1).item()


# =========================================================
# 7. Display Result
# =========================================================

print("\n========== PREDICTION ==========")

print("Predicted Digit:", prediction)

print("================================")