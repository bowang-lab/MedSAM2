import os
from torchvision.utils import save_image
import torch.nn.functional as F
import torch
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import generate_binary_structure


def get_boundary_from_masks(gt_masks: torch.Tensor, boundary_width: int = 3) -> torch.Tensor:
    """
    Generates boundary ground truth from instance segmentation masks with adjustable width.

    Args:
        gt_masks (torch.Tensor): The ground truth masks, expected to be a binary tensor of
                                 shape (B, 1, H, W) or (B, H, W).
        boundary_width (int): The desired width of the boundary in pixels. For a width W,
                              the boundary will extend roughly W/2 pixels inside and
                              W/2 pixels outside the original edge.

    Returns:
        torch.Tensor: The ground truth boundaries, a binary tensor of the same shape as input.
    """
    # Ensure input is a 4D tensor (B, 1, H, W) and on the CPU for numpy conversion
    input_shape = gt_masks.shape
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)  # Add channel dim if missing

    gt_masks_numpy = gt_masks.cpu().numpy().astype(bool)

    # --- MODIFICATION FOR ROBUST BOUNDARIES ---

    # 1. Use a solid, fully-connected structuring element.
    #    generate_binary_structure(rank, connectivity) creates a structure.
    #    rank=2 for 2D images.
    #    connectivity=2 means it's a solid 3x3 square (considers corners), leading to smoother results.
    footprint = generate_binary_structure(rank=2, connectivity=2)

    # To match the 4D input shape of the masks (B, C, H, W)
    footprint_4d = np.expand_dims(footprint, axis=(0, 1))

    # 2. Erode the masks by the specified width.
    #    The number of iterations directly controls the thickness of the inward part of the boundary.
    eroded_masks = binary_erosion(
        gt_masks_numpy,
        structure=footprint_4d,
        iterations=boundary_width
    )

    # The boundary is the difference between the original mask and the eroded mask.
    # This creates a boundary that is `boundary_width` pixels thick, extending *inside* the object.
    boundaries = np.logical_xor(gt_masks_numpy, eroded_masks)

    # Convert back to a PyTorch tensor on the original device
    return torch.from_numpy(boundaries.astype(np.float32)).to(gt_masks.device)

# def get_boundary_from_masks(gt_masks: torch.Tensor, boundary_width: int = 1) -> torch.Tensor:
#     """
#     Generates boundary ground truth from instance segmentation masks.

#     Args:
#         gt_masks (torch.Tensor): The ground truth masks, expected to be a binary tensor of
#                                  shape (B, 1, H, W) or (B, H, W).
#         boundary_width (int): The width of the boundary to generate.

#     Returns:
#         torch.Tensor: The ground truth boundaries, a binary tensor of the same shape as input.
#     """
#     # Ensure input is a 4D tensor (B, 1, H, W) and on the CPU for numpy conversion
#     input_shape = gt_masks.shape
#     if gt_masks.dim() == 3:
#         gt_masks = gt_masks.unsqueeze(1)  # Add channel dim if missing

#     gt_masks_numpy = gt_masks.cpu().numpy().astype(bool)

#     # Define the structuring element for erosion
#     # For a width of 1, a 3x3 cross is standard
#     footprint = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]])

#     # Erode the masks
#     eroded_masks = binary_erosion(gt_masks_numpy, structure=footprint, iterations=boundary_width)

#     # The boundary is the difference between the original mask and the eroded mask
#     boundaries = np.logical_xor(gt_masks_numpy, eroded_masks)

#     # Convert back to a PyTorch tensor on the original device
#     return torch.from_numpy(boundaries.astype(np.float32)).to(gt_masks.device)


# --- You should already have this function ---
# from .your_utils import get_boundary_from_masks

def save_boundary_visualization(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    original_images: torch.Tensor,  # We need the original images for context
    save_path: str,
    prefix: str = "boundary_viz",
    max_images: int = 4  # Only save a few images from the batch to avoid clutter
):
    """
    Saves a grid of images for visualizing boundary predictions against ground truth.

    The output grid will contain rows of:
    [Original Image, Ground Truth Mask, Ground Truth Boundary, Predicted Boundary Probabilities]

    Args:
        pred_logits (torch.Tensor): The raw logits from the boundary head. 
                                    Shape: (B, 1, H, W).
        gt_masks (torch.Tensor): The original ground truth masks (not boundaries yet).
                                 Shape: (B, 1, H, W).
        original_images (torch.Tensor): The corresponding original input images.
                                        Shape: (B, 3, H, W).
        save_path (str): The directory where the image will be saved.
        prefix (str): A prefix for the saved filename.
        max_images (int): The maximum number of samples from the batch to visualize.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Detach tensors from the computation graph and move to CPU
    pred_logits = pred_logits.detach().cpu()
    gt_masks = gt_masks.detach().cpu()
    original_images = original_images.detach().cpu()

    # Limit the number of images to save
    batch_size = pred_logits.shape[0]
    num_to_save = min(batch_size, max_images)

    pred_logits = pred_logits[:num_to_save]
    gt_masks = gt_masks[:num_to_save]
    original_images = original_images[:num_to_save]

    # --- Prepare the data for visualization ---

    # 1. Convert prediction logits to probability maps (0-1 range)
    pred_probs = torch.sigmoid(pred_logits)

    # 2. Generate ground truth boundaries from ground truth masks
    gt_boundaries = get_boundary_from_masks(gt_masks)

    # 3. Ensure all single-channel masks are repeated to 3 channels to be visualized in color
    # This makes the visualization grid look uniform.
    gt_masks_3c = gt_masks.repeat(1, 3, 1, 1)
    gt_boundaries_3c = gt_boundaries.repeat(1, 3, 1, 1)
    pred_probs_3c = pred_probs.repeat(1, 3, 1, 1)

    # 4. Normalize original images if they are not in the [0, 1] range
    # This is important if your dataloader applies normalization like z-score.
    # A simple min-max normalization for visualization purposes.
    img_min = original_images.min()
    img_max = original_images.max()
    if img_min < 0 or img_max > 1:
        original_images = (original_images - img_min) / (img_max - img_min)

    # 5. Stack all images together for a single grid
    # The list will be [img1, mask1, gt_b1, pred_b1, img2, mask2, ...]
    viz_list = []
    for i in range(num_to_save):
        viz_list.append(original_images[i])
        viz_list.append(gt_masks_3c[i])
        viz_list.append(gt_boundaries_3c[i])
        viz_list.append(pred_probs_3c[i])

    # Create the grid. nrow is the number of columns.
    # We have 4 columns: Original Image, GT Mask, GT Boundary, Pred Boundary
    grid = torch.stack(viz_list, dim=0)

    # Save the image
    filename = os.path.join(save_path, f"{prefix}_boundary_viz.png")
    save_image(grid, filename, nrow=4)  # 4 images per row

    print(f"Boundary visualization saved to {filename}")


def save_boundary_visualization_simple(
    pred_logits: torch.Tensor,
    gt_boundaries: torch.Tensor,
    save_path: str,
    prefix: str = "boundary_viz",
    epoch_num: int = 0,
    batch_idx: int = 0,
    max_images: int = 8  # Save a few more since they are smaller
):
    """
    Saves a grid of images for visualizing predicted boundaries against ground truth boundaries.

    The output grid will contain rows of:
    [Ground Truth Boundary 1, Predicted Boundary 1, Ground Truth Boundary 2, Predicted Boundary 2, ...]

    Args:
        pred_logits (torch.Tensor): The raw logits from the boundary head. 
                                    Shape: (B, 1, H, W).
        gt_boundaries (torch.Tensor): The ground truth boundary masks.
                                      Shape: (B, 1, H, W).
        save_path (str): The directory where the image will be saved.
        prefix (str): A prefix for the saved filename.
        epoch_num, batch_idx: For unique filenames.
        max_images (int): The maximum number of samples from the batch to visualize.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Detach tensors from the computation graph and move to CPU
    pred_logits = pred_logits.detach().cpu()
    gt_boundaries = gt_boundaries.detach().cpu()

    # Limit the number of images to save
    batch_size = pred_logits.shape[0]
    num_to_save = min(batch_size, max_images)

    pred_logits = pred_logits[:num_to_save]
    gt_boundaries = gt_boundaries[:num_to_save]

    # --- Prepare the data for visualization ---

    # 1. Convert prediction logits to probability maps (0-1 range)
    pred_probs = torch.sigmoid(pred_logits)

    # 2. Stack all images together for a single grid
    # The list will be [gt_b1, pred_b1, gt_b2, pred_b2, ...]
    viz_list = []
    for i in range(num_to_save):
        viz_list.append(gt_boundaries[i])
        viz_list.append(pred_probs[i])

    # Create the grid. nrow is the number of columns.
    # We have 2 images per sample (GT and Pred), let's make 4 pairs per row (8 columns)
    grid = torch.stack(viz_list, dim=0)

    # Save the image
    filename = os.path.join(save_path, f"{prefix}_epoch_{epoch_num}_batch_{batch_idx}.png")
    save_image(grid, filename, nrow=8)  # 8 images per row (4 pairs of GT/Pred)

    # Optional: print a confirmation message, but maybe not in the loss function
    # to avoid excessive logging.
    # print(f"Boundary visualization saved to {filename}")
