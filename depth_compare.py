import cv2
import torch
import numpy as np
from pathlib import Path
import depth_pro
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_model():
    """Load DepthPro model with bfloat16 precision."""
    model, transform = depth_pro.create_model_and_transforms(
        precision=torch.bfloat16, device=torch.device("mps"))
    return model, transform


def process_image(model, transform, image_path):
    """Process single image and return depth prediction."""
    # Read image
    frame = cv2.imread(str(image_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process first frame to get focal length
    image, _, f_px = depth_pro.load_rgb_from_numpy(frame)
    image = transform(image)

    # Run inference
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in meters

    return depth


def calculate_metrics(pred_depth, gt_depth):
    """Calculate depth estimation metrics."""
    # Convert to numpy for calculations
    pred = pred_depth.cpu().numpy()
    gt = gt_depth.cpu().numpy()

    # Calculate various metrics
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mae = np.mean(np.abs(pred - gt))
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    return {
        'rmse': rmse,
        'mae': mae,
        'abs_rel': abs_rel
    }


def visualize_comparison(pred_depth, gt_depth, save_path):
    """Visualize depth prediction comparison."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot predicted depth
    im1 = ax1.imshow(pred_depth.cpu().numpy(), cmap='plasma')
    ax1.set_title('Predicted Depth')
    plt.colorbar(im1, ax=ax1)

    # Plot ground truth depth
    im2 = ax2.imshow(gt_depth.cpu().numpy(), cmap='plasma')
    ax2.set_title('Ground Truth Depth')
    plt.colorbar(im2, ax=ax2)

    # Plot difference
    diff = (pred_depth - gt_depth).abs().cpu().numpy()
    im3 = ax3.imshow(diff, cmap='plasma')
    ax3.set_title('Absolute Difference')
    plt.colorbar(im3, ax=ax3)

    plt.savefig(save_path)
    plt.close()


def main():
    # Set up paths
    # Replace with actual path
    # Replace with actual path
    pred_dir = "../course_cpp/highres/original/"
    # Replace with actual path
    # gt_dir = "../course_cpp/highres/enhanced_images/"

    gt_dir = "../course_cpp/highres/darken_images/"
    output_dir = "output_dir"     # Replace with actual path
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    # Load model
    print("Loading model...")
    model, transform = load_model()

    # Get matching files from both directories
    import glob
    pred_files = glob.glob(os.path.join(pred_dir, "*.jpg"))
    gt_files = glob.glob(os.path.join(gt_dir, "*/enhanced.jpg"))
    #gt_files = glob.glob(os.path.join(gt_dir, "*.jpg"))
    # Match gt_files based on pred_files names
    matched_gt_files = []
    for pred_file in pred_files:
         pred_name = Path(pred_file).stem
         matching_gt = [gt for gt in gt_files if pred_name in Path(gt).parts]
         if matching_gt:
             matched_gt_files.append(matching_gt[0])
    gt_files = matched_gt_files
    # print(pred_files, gt_files)
    # Ensure we have matching files
    assert len(pred_files) == len(
        gt_files), f"{len(pred_files)} {len(gt_files)}"
    pred_files, gt_files = gt_files, pred_files
    # Process all image pairs
    metrics_list = []
    print("Processing images...")
    for pred_path, gt_path in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        # Process both images
        pred_path, gt_path = Path(pred_path), Path(gt_path)
        pred_depth = process_image(model, transform, pred_path)
        gt_depth = process_image(model, transform, gt_path)

        # Calculate metrics
        metrics = calculate_metrics(pred_depth, gt_depth)
        metrics_list.append(metrics)

        print(pred_path, gt_path, metrics)
        # Visualize results
        viz_path = output_dir / f"comparison_{pred_path.stem}.png"
        visualize_comparison(pred_depth, gt_depth, viz_path)

    # Calculate and print average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in metrics_list])
        for metric in metrics_list[0].keys()
    }

    print("\nAverage Metrics:")
    print(f"RMSE: {avg_metrics['rmse']:.4f}")
    print(f"MAE: {avg_metrics['mae']:.4f}")
    print(f"Absolute Relative Error: {avg_metrics['abs_rel']:.4f}")


if __name__ == "__main__":
    main()
