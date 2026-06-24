import os
import uuid
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_tools.core.utils.io import get_data_dir, get_local_path as ls_get_local_path

try:
    from label_studio_sdk.converter import brush
except ImportError:
    from label_studio_converter import brush

from inference import load_model, predict_mask

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "runs_deeplab",
    "runs",
    "train",
    "weights",
    "best_segmentation_model.pth",
)

LABEL_NAME = "person"

MODEL_VERSION = "deeplabv3-person-v1"
THRESHOLD = 0.5

DEFAULT_FROM_NAME = "brush"
DEFAULT_TO_NAME = "image"
DEFAULT_IMAGE_KEY = "image"


class DeepLabPersonBackend(LabelStudioMLBase):
    _shared_model = None
    _shared_device = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(os.environ.get("DEEPLAB_THRESHOLD", THRESHOLD))
        self.label_name = os.environ.get("DEEPLAB_LABEL", LABEL_NAME)
        self.model_version = os.environ.get("DEEPLAB_MODEL_VERSION", MODEL_VERSION)

    @classmethod
    def preload(cls, weights_path: Optional[str] = None):
        if cls._shared_model is not None:
            return cls._shared_model, cls._shared_device

        if weights_path is None:
            weights_path = os.path.abspath(
                os.environ.get("DEEPLAB_WEIGHTS_PATH", WEIGHTS_PATH)
            )

        print(f"Loading DeepLab weights from: {weights_path}")
        cls._shared_model, cls._shared_device = load_model(weights_path)
        print("DeepLab model loaded.")
        return cls._shared_model, cls._shared_device

    def _ensure_model(self):
        if self._shared_model is None:
            self.preload()
        self.model = self._shared_model
        self.device = self._shared_device

    def setup(self):
        pass

    def _parse_brush_config(self) -> Tuple[str, str, str]:
        from_name = None
        to_name = None
        image_key = None

        for tag_name, tag_info in self.parsed_label_config.items():
            tag_type = tag_info.get("type", "").lower()
            if tag_type == "brushlabels":
                from_name = tag_name
                to_name = tag_info["to_name"][0]
            elif tag_type == "image":
                image_key = tag_info.get("value") or tag_info.get("valueType")
                if image_key and image_key.startswith("$"):
                    image_key = image_key[1:]

        return (
            from_name or DEFAULT_FROM_NAME,
            to_name or DEFAULT_TO_NAME,
            image_key or DEFAULT_IMAGE_KEY,
        )

    def _find_local_upload(self, image_url: str) -> Optional[str]:
        path_part = urlparse(image_url).path if "://" in image_url else image_url
        parts = [p for p in path_part.split("/") if p]

        if "upload" not in parts:
            return None

        idx = parts.index("upload")
        if len(parts) <= idx + 2:
            return None

        project_id = parts[idx + 1]
        filename = unquote(parts[idx + 2])
        upload_dir = os.path.join(get_data_dir(), "media", "upload", project_id)

        candidate = os.path.join(upload_dir, filename)
        if os.path.isfile(candidate):
            return candidate

        prefix = filename.split("-")[0]
        if os.path.isdir(upload_dir):
            for name in os.listdir(upload_dir):
                if name.startswith(prefix):
                    full = os.path.join(upload_dir, name)
                    if os.path.isfile(full):
                        return full

        return None

    def _get_image_path(self, task: Dict, image_key: str) -> str:
        image_url = task["data"][image_key]
        task_id = task.get("id")

        local_path = self._find_local_upload(image_url)
        if local_path:
            return local_path

        hostname = (
            self.hostname
            or os.environ.get("LABEL_STUDIO_URL", "")
            or os.environ.get("LABEL_STUDIO_HOST", "")
            or "http://localhost:8080"
        )
        access_token = (
            self.access_token
            or os.environ.get("LABEL_STUDIO_API_KEY", "")
            or os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", "")
        )

        path = ls_get_local_path(
            image_url,
            hostname=hostname if hostname.endswith("/") else hostname + "/",
            access_token=access_token,
            task_id=task_id,
            download_resources=True,
        )
        path = unquote(path)
        if os.path.isfile(path):
            return path

        raise FileNotFoundError(
            f"Image not found: {image_url}\n"
            f"Tried path: {path}\n"
            "Set LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY if files are stored in Label Studio."
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        self._ensure_model()
        from_name, to_name, image_key = self._parse_brush_config()
        predictions = []

        for task in tasks:
            image_path = self._get_image_path(task, image_key)
            mask, (width, height), score = predict_mask(
                self.model,
                image_path,
                self.device,
                threshold=self.threshold,
            )

            if mask.max() == 0:
                predictions.append(
                    {
                        "model_version": self.model_version,
                        "score": 0.0,
                        "result": [],
                    }
                )
                continue

            mask_binary = (mask > 127).astype(np.uint8) * 255
            rle = brush.mask2rle(mask_binary)

            predictions.append(
                {
                    "model_version": self.model_version,
                    "score": float(score) if score is not None else 0.0,
                    "result": [
                        {
                            "id": str(uuid.uuid4())[:8],
                            "from_name": from_name,
                            "to_name": to_name,
                            "type": "brushlabels",
                            "origin": "prediction",
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "value": {
                                "format": "rle",
                                "rle": rle,
                                "brushlabels": [self.label_name],
                            },
                        }
                    ],
                }
            )

        return predictions
