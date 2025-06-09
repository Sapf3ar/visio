import cv2
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from loguru import logger
import depth_pro
from ollama import chat, ChatResponse
from pydantic import BaseModel
from isegm_inf import load_model, run_mobilesam_inference


class Click(BaseModel):
    x: int
    y: int
    name: str


predictor = load_model()


def infer_vlm(msg: str = "Return an xy coordinate of a center of the closest object in an image and a name of this object which point belongs to. Resolution of the image is (896 , 896). Output format is: {'x': _, 'y':_}.",
              # vlm_client=client,
              image: np.ndarray | None = None,
              depth: np.ndarray | None = None):
    response: ChatResponse = chat(model='gemma3:4b',
                                  messages=[
                                      {
                                          'role': 'user',
                                          'content': msg,
                                          'images': [cv2.imencode('.jpg', image)[1].tobytes()]
                                      }
                                  ],
                                  format=Click.model_json_schema())
    return Click.model_validate_json(response['message']['content'])


# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(
    precision=torch.bfloat16, device=torch.device("mps"))


# Load and preprocess an image.

cap = cv2.VideoCapture(0)  # 0=built-in camera, 1=external
f_px = None
while True:
    ret, frame = cap.read()

    if f_px == None:
        image, _, f_px = depth_pro.load_rgb_from_numpy(frame)
        image = transform(image)
    else:
        image = transform(frame)
    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    # Depth in [m].
    depth = prediction["depth"]
    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )
    color_depth = (inverse_depth_normalized * 255).cpu().numpy().astype(
        np.uint8
    )
    print(color_depth.shape)
    # Draw horizontal lines
    frame_height, frame_width = frame.shape[:2]
    for i in range(1, 9):
        y = int(frame_height * i / 9)
        cv2.line(frame, (0, y), (frame_width, y), (0, 255, 0), 2)

    # Draw vertical lines
    for i in range(1, 9):
        x = int(frame_width * i / 9)
        cv2.line(frame, (x, 0), (x, frame_height), (0, 255, 0), 2)
    click: Click = infer_vlm(depth=np.stack(
        [color_depth, color_depth, color_depth], axis=-1), image=frame)
    print(click)
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    # Rescale click coordinates from 896x896 to 1080x1920 resolution
    if len(str(click.x)) > 3:
        click.x = int(str(click.x)[:3])
    if len(str(click.y)) > 3:
        click.y = int(str(click.y)[:3])
    click.x = int((click.x / 896) * 1920)
    click.y = int((click.y / 896) * 1080)
    masks, _, _ = run_mobilesam_inference(
        predictor, frame, np.array([click.x, click.y])[None, ...])
    logger.debug(f"{masks.max()}")

    cv2.circle(color_depth, (click.x, click.y), 10, (255, 0, 0), 2)
    # cv2.imshow('Webcam', color_depth)
    # cv2.imshow('Webcam', masks.astype(np.uint8) * 255)
    # Blend depth map with segmentation mask
    masked_depth = cv2.addWeighted(
        color_depth, 0.7, masks.astype(np.uint8) * 255, 0.3, 0)

    # Add text overlay for object name
    cv2.putText(masked_depth, click.name, (click.x + 15, click.y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    # Blend masked_depth with color frame
    blended_frame = cv2.addWeighted(frame, 0.5, cv2.cvtColor(
        masked_depth, cv2.COLOR_GRAY2BGR), 0.5, 0)
    cv2.imshow('Blended View', blended_frame)
    # Display blended result
    # cv2.imshow('Depth + Mask', masked_depth)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
