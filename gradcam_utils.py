import torch 
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        #to store activations and gradients
        self.activation = None
        self.gradients = None

        ##HOOK THE FORWARD AND BACKWARD PASSES
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        # grad_output is always a tuple
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: single image tensor [1, C, H, W]
        class_idx: class index for which Grad-CAM is computed (default = predicted class)
        """

        # forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):  # <-- FIX
            output = output[0]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # backward pass: get gradients wrt chosen class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients -> weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # weighted sum of activation
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """
    img: original image (H,W,3) in [0,1]
    mask: CAM heatmap (H_feat,W_feat) or (H_feat,W_feat,1)
    """
    # 1. Resize mask to match image
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # 2. Normalize to [0,1]
    mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)

    # 3. Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 4. Overlay heatmap on image
    overlay = np.float32(heatmap) / 255
    result = overlay * alpha + img * (1 - alpha)
    result = np.clip(result, 0, 1)

    plt.imshow(result)
    plt.axis("off")
    return result