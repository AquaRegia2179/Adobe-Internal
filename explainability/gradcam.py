import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_gradcam_heatmap(model, input_tensor, target_layer):
    device = next(model.parameters()).device  

    cam = GradCAM(model=model, target_layers=[target_layer])

    input_tensor = input_tensor.to(device)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0] 
    return grayscale_cam

def overlay_heatmap_on_image(heatmap, original_image):
    if original_image.max() > 1:
        img = original_image.astype(np.float32) / 255.0
    else:
        img = original_image

    heatmap_img = show_cam_on_image(img, heatmap, use_rgb=True)
    return heatmap_img
