import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from gradcam_utils import GradCAM, show_cam_on_image
from predictor import PneumoniaPredictor, build_model
import numpy as np
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model and load weights
model = build_model(num_classes=2).to(DEVICE)

# Load the trained weights (update path if needed)
ckpt_path = r"E:\PNEUMONIA_APP\model.pth"
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()  # Set to evaluation mode

print("✅ Model loaded! Best Val AUC =", ckpt.get("val_auc", "unknown"))

# Initialize GradCAM and predictor
gradcam = GradCAM(model, model.layer4[-1])
predictor = PneumoniaPredictor(model, gradcam, device=DEVICE)

def gradio_fn(img):
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img).convert("RGB")
    
    # Transform and add batch dimension
    img_tensor = predictor.transform(pil_img).unsqueeze(0).to(predictor.device)

    # Prediction
    with torch.no_grad():
        outputs = predictor.model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    # Grad-CAM
    cam = predictor.gradcam.generate(img_tensor)
    img_np = np.array(pil_img.resize((224,224))) / 255.0

    # Overlay heatmap
    heatmap = show_cam_on_image(img_np, cam)

    return (
        {predictor.class_names[0]: float(probs[0][0]), 
         predictor.class_names[1]: float(probs[0][1])},
        heatmap
    )

demo = gr.Interface(
    fn=gradio_fn,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=[gr.Label(label="Prediction"), gr.Image(label="Grad-CAM")],
    title="Pneumonia Detection with Grad-CAM"
)

demo.launch()