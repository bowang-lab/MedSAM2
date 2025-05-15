import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from skimage import measure

from sam2.build_sam import build_sam2_video_predictor_npz

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MedSAM2 3D Inference for Multi-class Segmentation')
    
    parser.add_argument('--ct', type=str, required=True,
                        help='Path to CT volume (.nii/.nii.gz)')
    parser.add_argument('--mask', type=str, required=True,
                        help='Path to multi-class mask volume (.nii/.nii.gz)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='MedSAM-2 checkpoint path')
    parser.add_argument('--cfg', type=str, default="configs/sam2.1_hiera_t512.yaml",
                        help='Model config file')
    parser.add_argument('--hu_min', type=float, default=-500,
                        help='Minimum HU value for windowing (default: -500)')
    parser.add_argument('--hu_max', type=float, default=1500,
                        help='Maximum HU value for windowing (default: 1500)')
    parser.add_argument('--out_dir', type=str, default='./results',
                        help='Output directory (default: ./results)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size to resize images (default: 512)')
    
    return parser.parse_args()


def window_ct_volume(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Apply HU windowing to CT volume."""
    volume_windowed = np.clip(volume, hu_min, hu_max)
    volume_normalized = (volume_windowed - hu_min) / (hu_max - hu_min) * 255.0
    return np.uint8(volume_normalized)


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert binary mask to bounding box coordinates.
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max)
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return None
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    return (x_min, y_min, x_max, y_max)


def resize_grayscale_to_rgb(array: np.ndarray, image_size: int) -> np.ndarray:
    """Resize 3D grayscale array to RGB and resize to target size.
    
    Args:
        array: Input array of shape (D, H, W)
        image_size: Target size for width and height
        
    Returns:
        Array of shape (D, 3, image_size, image_size)
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    
    return resized_array


def collect_slice_metadata(ct_path: str, mask_path: str, args: argparse.Namespace) -> pd.DataFrame:
    """Collect metadata for all slices containing objects.
    
    Returns:
        DataFrame with slice information
    """
    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)
    
    ct_volume = sitk.GetArrayFromImage(ct_image)
    mask_volume = sitk.GetArrayFromImage(mask_image)
    
    # Apply HU windowing
    ct_volume_windowed = window_ct_volume(ct_volume, args.hu_min, args.hu_max)
    
    # Get spacing
    spacing = ct_image.GetSpacing()  # (x, y, z)
    
    # Get file stems
    ct_stem = Path(ct_path).stem.replace('.nii', '')
    mask_stem = Path(mask_path).stem.replace('.nii', '')
    
    metadata = []
    
    # Scan through slices
    for slice_idx in tqdm(range(mask_volume.shape[0]), desc="Collecting slice metadata"):
        slice_mask = mask_volume[slice_idx]
        
        # Get unique labels (excluding background)
        unique_labels = np.unique(slice_mask)
        unique_labels = unique_labels[unique_labels > 0]
        
        # Process each label
        for label in unique_labels:
            label_mask = (slice_mask == label).astype(np.uint8)
            bbox = mask_to_bbox(label_mask)
            
            if bbox is not None:
                metadata.append({
                    'ct_file_name': ct_stem,
                    'mask_file_name': mask_stem,
                    'slice_index': slice_idx,
                    'bounding_boxes': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    'spacing_mm_px': f"{spacing[0]:.6f},{spacing[1]:.6f},{spacing[2]:.6f}",
                    'image_size': f"{args.image_size},{args.image_size}",
                    'dicom_window': f"{args.hu_min},{args.hu_max}",
                    'label': int(label)
                })
    
    return pd.DataFrame(metadata)


def get_largest_connected_component(segmentation: np.ndarray) -> np.ndarray:
    """Get the largest connected component from a binary segmentation."""
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc.astype(np.uint8)


def calculate_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Dice coefficient between prediction and target."""
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def run_bbox_inference(ct_path: str, mask_path: str, case_info: pd.DataFrame, 
                      predictor: Any, args: argparse.Namespace) -> pd.DataFrame:
    """Run bounding box driven inference for all cases.
    
    Returns:
        DataFrame with segmentation results
    """
    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)
    
    ct_volume = sitk.GetArrayFromImage(ct_image)
    mask_volume = sitk.GetArrayFromImage(mask_image)
    
    # Apply HU windowing
    ct_volume_windowed = window_ct_volume(ct_volume, args.hu_min, args.hu_max)
    
    # Prepare volume for SAM
    img_resized = resize_grayscale_to_rgb(ct_volume_windowed, args.image_size)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    
    # Normalize
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    img_resized = (img_resized - img_mean) / img_std
    
    # Video dimensions
    video_height, video_width = ct_volume.shape[1], ct_volume.shape[2]
    
    results = []
    
    # Process each row in case_info
    for idx, row in tqdm(case_info.iterrows(), total=len(case_info), desc="Running inference"):
        slice_idx = row['slice_index']
        label = row['label']
        bbox_str = row['bounding_boxes']
        bbox = [int(x) for x in bbox_str.split(',')]
        
        # Initialize predictor state
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            
            # Add bounding box
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                box=bbox,
            )
            
            # Forward propagation
            pred_volume = np.zeros(ct_volume.shape, dtype=np.uint8)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                pred_volume[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            
            predictor.reset_state(inference_state)
            
            # Backward propagation
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=slice_idx,
                obj_id=1,
                box=bbox,
            )
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                pred_volume[out_frame_idx] |= (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            
            predictor.reset_state(inference_state)
        
        # Post-process: get largest connected component
        if np.max(pred_volume) > 0:
            pred_volume = get_largest_connected_component(pred_volume)
        
        # Calculate Dice score with ground truth
        gt_volume = (mask_volume == label).astype(np.uint8)
        dice_score = calculate_dice(pred_volume, gt_volume)
        
        # Save prediction
        output_name = f"{row['ct_file_name']}_s{slice_idx}_cls{label}_pred.nii.gz"
        output_path = os.path.join(args.out_dir, output_name)
        
        pred_image = sitk.GetImageFromArray(pred_volume)
        pred_image.CopyInformation(ct_image)
        sitk.WriteImage(pred_image, output_path)
        
        # Record results
        results.append({
            'ct_file_name': row['ct_file_name'],
            'slice_index': slice_idx,
            'label': label,
            'dice_score': dice_score,
            'output_file': output_name
        })
    
    return pd.DataFrame(results)


def save_results(case_info: pd.DataFrame, seg_info: pd.DataFrame, args: argparse.Namespace):
    """Save case information and segmentation results."""
    case_info_path = os.path.join(args.out_dir, 'case_info.csv')
    seg_info_path = os.path.join(args.out_dir, 'segmentation_info.csv')
    
    case_info.to_csv(case_info_path, index=False)
    seg_info.to_csv(seg_info_path, index=False)
    
    print(f"Case information saved to: {case_info_path}")
    print(f"Segmentation results saved to: {seg_info_path}")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize predictor
    print("Loading MedSAM-2 model...")
    predictor = build_sam2_video_predictor_npz(args.cfg, args.ckpt)
    
    # Step 1: Collect slice metadata
    print("Collecting slice metadata...")
    case_info = collect_slice_metadata(args.ct, args.mask, args)
    
    if len(case_info) == 0:
        print("No objects found in the mask volume.")
        return
    
    # Save case info
    case_info_path = os.path.join(args.out_dir, 'case_info.csv')
    case_info.to_csv(case_info_path, index=False)
    print(f"Case information saved to: {case_info_path}")
    
    # Step 2: Run bounding box inference
    print("Running bounding box driven inference...")
    seg_info = run_bbox_inference(args.ct, args.mask, case_info, predictor, args)
    
    # Step 3: Save all results
    save_results(case_info, seg_info, args)
    
    # Print summary
    print("\nInference complete!")
    print(f"Processed {len(case_info)} slices with objects")
    print(f"Average Dice score: {seg_info['dice_score'].mean():.4f}")


if __name__ == "__main__":
    main()