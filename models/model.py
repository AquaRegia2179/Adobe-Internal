import torch
import torch.nn as nn
from torchvision import models

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def load_trained_model(weight_path, num_classes=2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)

    state = torch.load(weight_path, map_location=device)

    try:
        model.backbone.load_state_dict(state, strict=False)
    except:
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()
    return model
