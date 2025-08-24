import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from gradcam_utils import GradCAM, show_cam_on_image
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class PneumoniaPredictor:
    def __init__(self, model, gradcam, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.gradcam = gradcam
        self.device = device

        # same preprocessing as training
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.class_names = ['Normal', 'Pneumonia']

    def predict(self, img_path, show=True):
        # load image
        pil_img = Image.open(img_path).convert("RGB")

        # preprocess
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        pred_label = self.class_names[pred_class]

        # Undo normalization for viz
        img_np = np.array(pil_img.resize((224,224))) / 255.0

        # grad-cam
        cam = self.gradcam.generate(img_tensor)

        if show:
            plt.figure(figsize=(10,4))

            plt.subplot(1,2,1)
            plt.imshow(img_np)
            plt.title("Uploaded X-ray")
            plt.axis("off")

            plt.subplot(1,2,2)
            show_cam_on_image(img_np, cam)
            plt.title(f"Grad-CAM\nPred: {pred_label} ({confidence*100:.1f}%)")
            plt.axis("off")

            plt.show()

        return pred_label, confidence