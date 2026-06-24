import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def create_deeplabv3_model(num_classes: int = 1):
    model = deeplabv3_resnet101(weights=None, progress=False)
    model.classifier = DeepLabHead(2048, num_classes)
    return model


def load_model(weights_path: str, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = create_deeplabv3_model(num_classes=1)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)

    model_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")

    model.to(device)
    model.eval()
    return model, device


def keep_largest_component(mask, use_probability=False, probability_mask=None):
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 127).astype(np.uint8) * 255
    else:
        mask_uint8 = mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask_uint8, None

    if use_probability and probability_mask is not None:
        max_prob = -1.0
        best_label = 1
        for label in range(1, num_labels):
            component_mask = labels == label
            if component_mask.sum() > 0:
                avg_prob = float(probability_mask[component_mask].mean())
                if avg_prob > max_prob:
                    max_prob = avg_prob
                    best_label = label
        return (labels == best_label).astype(np.uint8) * 255, max_prob

    largest_area = 0
    best_label = 1
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            best_label = label

    result_mask = (labels == best_label).astype(np.uint8) * 255
    selected_prob = None
    if probability_mask is not None:
        component_mask = labels == best_label
        selected_prob = float(probability_mask[component_mask].mean())
    return result_mask, selected_prob


def refine_mask_edges(mask, kernel_size=3, iterations=1):
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 127).astype(np.uint8) * 255
    else:
        mask_uint8 = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask_smooth = cv2.medianBlur(mask_opened, 5)
    _, mask_final = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
    return mask_final


def predict_mask(
    model,
    image_path: str,
    device,
    target_size=(512, 512),
    threshold: float = 0.5,
    refine_edges: bool = True,
    keep_largest_only: bool = True,
    use_probability: bool = True,
):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image_resized = image.resize(target_size, Image.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image_resized).astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    image_tensor = (image_tensor - mean) / std

    with torch.no_grad():
        output = model(image_tensor)["out"]
        mask_prob = torch.sigmoid(output).squeeze().cpu().numpy()

    mask_binary = (mask_prob > threshold).astype(np.uint8) * 255
    mask_resized = Image.fromarray(mask_binary).resize(original_size, Image.NEAREST)
    mask_array = np.array(mask_resized)

    mask_prob_resized = None
    if keep_largest_only and use_probability:
        mask_prob_resized = Image.fromarray((mask_prob * 255).astype(np.uint8)).resize(
            original_size, Image.BILINEAR
        )
        mask_prob_resized = np.array(mask_prob_resized).astype(np.float32) / 255.0

    selected_probability = None
    if keep_largest_only:
        mask_array, selected_probability = keep_largest_component(
            mask_array,
            use_probability=use_probability,
            probability_mask=mask_prob_resized,
        )

    if refine_edges:
        mask_array = refine_mask_edges(mask_array, kernel_size=3, iterations=1)

    return mask_array, original_size, selected_probability
