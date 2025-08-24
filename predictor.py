import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from gradcam_utils import GradCAM, show_cam_on_image

def load_model(model_path="pneumonia_model.pth", device=None):
    """
    Load the trained Pneumonia classification model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model architecture (must match training!)
    # Replace this with your actual model definition
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal vs Pneumonia

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, device

def preprocess_image(image: Image.Image):
    """
    Apply preprocessing to input image.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # must match training input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # [1, C, H, W]

def PneumoniaPredictor(image: Image.Image, model, device):
    """
    Run model inference + Grad-CAM visualization.
    """
    input_tensor = preprocess_image(image).to(device)

    # Run Grad-CAM on last conv layer
    target_layer = list(model.layer4.children())[-1]  # last ResNet block
    cam = GradCAM(model, target_layer)
    mask = cam.generate(input_tensor)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    classes = ["Normal", "Pneumonia"]
    label = f"{classes[pred_class]} ({probs[pred_class]:.2f})"

    # Convert PIL -> np for overlay
    img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_np, mask)

    return label, overlay
