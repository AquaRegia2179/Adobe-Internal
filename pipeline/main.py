import torch
import cv2
import sys
import os
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import load_trained_model
from explainability.gradcam import get_gradcam_heatmap
from explainability.roi_extraction import extract_roi
from explainability.clip_labelling import CLIPArtifactDetector

#TODO:
# Make the roi extraction more stable/better
# Expand the list of reasonings

#NOTE: The model being used rn is resnet50.

def analyze_image(image_path, model_weights="models/weights/best_resnet50.pth", save_results=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(weight_path=model_weights,num_classes=2)


    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0


    from torchvision import transforms



    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    img_resized = cv2.resize(img_rgb, (224, 224))
    tensor = transform(img_resized).unsqueeze(0).to(device)



    heatmap = get_gradcam_heatmap(model, tensor, target_layer = model.backbone.layer4[-1])

    print(heatmap.shape)
    import matplotlib.pyplot as plt

    heatmap_uint8 = np.uint8(255 * heatmap)


    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


    overlayed_image = cv2.addWeighted(img_resized, 0.6, heatmap_rgb, 0.4, 0)


    
    plt.figure(figsize=(7, 7))
    plt.imshow(overlayed_image)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')  
    plt.show()


    # img_float = img_rgb.astype(np.float32) / 255.0

    # heatmap_on_image = show_cam_on_image(img_float, heatmap, use_rgb=True)

    # heatmap_bgr = cv2.cvtColor(heatmap_on_image, cv2.COLOR_RGB2BGR)

    # if save_results:
    #     cv2.imwrite("reports/explainability/gradcam_overlay.jpg", heatmap_bgr)

    
    
    #print("Debug 1")
    rois, boxes = extract_roi(heatmap, img)

    if not rois:
        print("⚠ No significant artifact regions found.")
        return

    clip_detector = CLIPArtifactDetector()
    #print("Debug 2")

    results = []
    for i, roi in enumerate(rois):
        artifacts = clip_detector.classify_artifacts(roi, top_k=3)
        results.append({
            "roi_index": i,
            "box": boxes[i],  # (x, y, w, h)
            "artifacts": artifacts
        })

        print(f"\n ROI #{i} at {boxes[i]}:")
        for label, score in artifacts:
            print(f"   - {label} (confidence: {score:.3f})")

        if save_results:
            roi.save(f"reports/explainability/roi_{i}.png")

    if save_results:
        heatmap_img = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)
        heatmap_path = "reports/explainability/heatmap.jpg"
        cv2.imwrite(heatmap_path, heatmap_img)
        #print(Debug 3)
    return results


if __name__ == "__main__":
    image_path = "data/test/FAKE/0 (5).jpg" 

    # image = cv2.imread(image_path)
    # cv2.imshow('Loaded Image', image)

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    analyze_image(image_path)
