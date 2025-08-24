import gradio as gr
import torch
from predictor import PneumoniaPredictor
from gradcam_utils import GradCAM, show_cam_on_image

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pth", map_location=device)  # replace with your trained model path
model.eval()

# Grad-CAM setup (last conv layer)
gradcam = GradCAM(model, target_layer="layer4")  # change if your model has a different last conv block

# Predictor instance
predictor = PneumoniaPredictor(model, gradcam, device=device)

def gradio_fn(image):
    pred_label, confidence, img_np, cam = predictor.predict(image, show=False)

    # overlay CAM
    heatmap = show_cam_on_image(img_np, cam)

    return pred_label, confidence, heatmap

demo = gr.Interface(
    fn=gradio_fn,
    inputs=gr.Image(type="filepath", label="Upload Chest X-ray"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap")
    ],
    title="Pneumonia Detection App",
    description="Upload a chest X-ray and the model will predict Normal vs Pneumonia with Grad-CAM visualization."
)

if __name__ == "__main__":
    demo.launch()
