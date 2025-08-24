import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook for forward pass
        self.fwd_hook = target_layer.register_forward_hook(self.save_activation)
        # Hook for backward pass (full_backward_hook to avoid warnings)
        self.bwd_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple → take the first element
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        # Forward
        scores = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(scores, dim=1).item()

        # Backward
        target = scores[:, class_idx]
        target.backward()

        # Compute Grad-CAM
        grads = self.gradients  # [B, C, H, W]
        activs = self.activations  # [B, C, H, W]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        cam = torch.sum(weights * activs, dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0,1]
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        # Upsample CAM to match input size
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        return cam.squeeze().cpu().numpy()


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5, show: bool = False):
    """
    Overlay Grad-CAM mask on image.

    Args:
        img (np.ndarray): Original image (H,W,3) in [0,1].
        mask (np.ndarray): Grad-CAM mask (H,W) in [0,1].
        alpha (float): Transparency.
        show (bool): Whether to display with matplotlib.

    Returns:
        np.ndarray: Overlay image (H,W,3) in uint8.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(255 * img)
    overlay = np.clip(overlay * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Grad-CAM Overlay")
        plt.show()

    return overlay
