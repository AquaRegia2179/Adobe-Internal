import numpy as np
import cv2
from PIL import Image

def extract_roi(heatmap, original_image, threshold=0.1, min_area=200):
    H,W = original_image.shape[:2]
    count = 0
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # for i in heatmap_resized:
    #     for j in i:
    #         if j > threshold:
    #             count+=1

    # print(f"The count is {count}")


    mask = (heatmap_resized > threshold).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    boxes = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if w*h < min_area:
            continue

        crop = original_image[y:y+h, x:x+w]

        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        rois.append(Image.fromarray(crop))
        boxes.append((x,y,w,h))
        
    # print("Shape of rois is")
    # print(len(rois))
    return rois, boxes

