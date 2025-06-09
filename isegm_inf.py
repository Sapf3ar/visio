import numpy as np
import cv2
import torch
from loguru import logger

from mobile_sam import sam_model_registry, SamPredictor


def load_model() -> SamPredictor:
    model_type: str = "vit_t"
    sam_checkpoint: str = "/Users/sapfaer/projects/robs/cw/MobileSAM/weights/mobile_sam.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    predictor = SamPredictor(mobile_sam)
    return predictor


def run_mobilesam_inference(predictor: SamPredictor, image: np.ndarray, input_point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MobileSAM inference on input image using provided model.
    Returns tuple of (masks, scores, logits)"""

    image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize predictor with model
    predictor.set_image(image_rgb)

    # Use center point as prompt
    h, w = image.shape[:2]
    # input_point: np.ndarray = inpu
    input_label: np.ndarray = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    return masks[scores.argmax(), ...], scores, logits
