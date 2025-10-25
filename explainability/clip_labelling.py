import torch
import numpy as np
from transformers import CLIPProcessor , CLIPModel
from PIL import Image
import os
import cv2 as cv2

#TODO:
#   Update the list of artifact candidates
ARTIFACT_CANDIDATES = [
    "deformed hands with extra fingers",
    "mangled hands with fused fingers",
    "six fingers on one hand",
    "asymmetrical eyes or mismatched pupils",
    "unnatural or waxy skin texture",
    "plastic-looking skin",
    "blurry or distorted face",
    "garbled or unnatural teeth",
    "mismatched or poorly formed ears",
    "illogical body parts or extra limbs",
    "garbled or unreadable text",
    "nonsense writing or symbols",
    "warped or distorted letters",
    "a strange repeating pattern",
    "an unnatural moiré pattern",
    "smeary or watery texture",
]


class CLIPArtifactDetector:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.artifact_texts = ARTIFACT_CANDIDATES
        self.text_embeds = self._embed_texts(self.artifact_texts)

    def _embed_texts(self, text_list):
        inputs =  self.processor(text = text_list, return_tensors = "pt", padding = True).to(self.device)

        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features/text_features.norm(dim = 1, keepdim=True)

        return text_features
    

    def classify_artifacts(self, roi, top_k=3):
        try:
            if isinstance(roi, np.ndarray):
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = Image.fromarray(roi)

            inputs = self.processor(images=roi, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                text_inputs = self.processor(
                    text=self.artifact_texts, return_tensors="pt", padding=True
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).squeeze(0)  
                probs = torch.softmax(similarity, dim=0)

                top_probs, top_indices = probs.topk(top_k)
                results = [
                    (self.artifact_texts[i], float(top_probs[j].item()))
                    for j, i in enumerate(top_indices)
                ]

                return results

        except Exception as e:
            print(f"Error in classify_artifacts: {e}")
            return [] 