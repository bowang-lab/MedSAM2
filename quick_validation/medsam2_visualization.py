#!/usr/bin/env python3
"""
MedSAM2 Visualization Script
Visualize segmentation results at different percentile slices

This script loads the segmentation results from MedSAM2 and creates
visualizations showing the original image and segmentation overlay
at the 25th percentile, key slice, and 75th percentile.
"""

import os
import re
import argparse
from os.path import join, basename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    Show mask on the image
    
    Args:
        mask: numpy.ndarray - mask of the image
        ax: matplotlib.axes.Axes - axes to plot the mask
        mask_color: numpy.ndarray - color of the mask
        alpha: float - transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    Show bounding box on the image
    
    Args:
        box: numpy.ndarray - bounding box coordinates
        ax: matplotlib.axes.Axes - axes to plot the bounding box
        edgecolor: str - color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, 
                              facecolor=(0,0,0,0), lw=2))


class MedSAM2Visualizer:
    """MedSAM2 visualization class for CT lesion segmentation results"""
    
    def __init__(self, pred_save_dir, path_DL_info, seg_info_csv):
        """
        Initialize the MedSAM2 visualizer
        
        Args:
            pred_save_dir: Directory containing saved prediction results
            path_DL_info: Path to the CSV file containing CT lesion info
            seg_info_csv: Path to the CSV file containing segmentation info
        """
        self.pred_save_dir = pred_save_dir
        self.path_DL_info = path_DL_info
        
        # Load dataset info
        self.DL_info = pd.read_csv(self.path_DL_info)
        self.seg_info = pd.read_csv(seg_info_csv)
        
        # Create visualization directory
        self.vis_save_dir = join(self.pred_save_dir, 'visualizations')
        os.makedirs(self.vis_save_dir, exist_ok=True)
    
    def preprocess_image(self, nii_image_data, lower_bound, upper_bound):
        """
        Preprocess the CT image data with windowing
        
        Args:
            nii_image_data: Raw CT image data
            lower_bound: Lower bound for windowing
            upper_bound: Upper bound for windowing
            
        Returns:
            Preprocessed image data
        """
        nii_image_data_pre = np.clip(nii_image_data, lower_bound, upper_bound)
        nii_image_data_pre = (nii_image_data_pre - np.min(nii_image_data_pre)) / \
                            (np.max(nii_image_data_pre) - np.min(nii_image_data_pre)) * 255.0
        return np.uint8(nii_image_data_pre)
    
    def visualize_single_case(self, seg_info_row):
        """
        Visualize a single segmentation case
        
        Args:
            seg_info_row: Row from the segmentation info dataframe
        """
        # Extract information from the segmentation filename
        seg_filename = seg_info_row['nii_name']
        key_slice_idx = int(seg_info_row['key_slice_index'])
        dicom_windows = seg_info_row['DICOM_windows']
        
        # Parse the original filename
        base_name = seg_filename.replace(f'_k{key_slice_idx}_mask.nii.gz', '')
        original_img_path = join(self.pred_save_dir, f'{base_name}_img.nii.gz')
        mask_path = join(self.pred_save_dir, seg_filename)
        
        if not os.path.exists(original_img_path) or not os.path.exists(mask_path):
            print(f"Files not found for {seg_filename}")
            return
        
        # Load the image and mask
        nii_image = sitk.ReadImage(original_img_path)
        nii_image_data = sitk.GetArrayFromImage(nii_image)
        
        nii_mask = sitk.ReadImage(mask_path)
        segs_3D = sitk.GetArrayFromImage(nii_mask)
        
        # Get windowing parameters
        lower_bound, upper_bound = dicom_windows.split(',')
        lower_bound, upper_bound = float(lower_bound), float(upper_bound)
        
        # Preprocess the image
        img_3D_ori = self.preprocess_image(nii_image_data, lower_bound, upper_bound)
        
        # Get slice range from original dataset info
        case_name_match = re.findall(r'(\d{6}_\d{2}_\d{2})', base_name)
        range_suffix_match = re.findall(r'\d{3}-\d{3}', base_name)
        
        if not case_name_match or not range_suffix_match:
            print(f"Could not parse filename: {base_name}")
            return
        
        case_name = case_name_match[0]
        range_suffix = range_suffix_match[0]
        slice_range = range_suffix.split('-')
        slice_range = [str(int(s)) for s in slice_range]
        slice_range_str = ', '.join(slice_range)
        
        # Find the corresponding case info
        case_df = self.DL_info[
            self.DL_info['File_name'].str.contains(case_name) & 
            self.DL_info['Slice_range'].str.contains(slice_range_str)
        ].copy()
        
        if case_df.empty:
            print(f"No matching case info found for {base_name}")
            return
        
        row = case_df.iloc[0]
        slice_range = row['Slice_range']
        slice_idx_start, slice_idx_end = slice_range.split(',')
        slice_idx_start, slice_idx_end = int(slice_idx_start), int(slice_idx_end)
        
        # Calculate key slice offset
        key_slice_idx_offset = key_slice_idx - slice_idx_start
        
        # Calculate percentile slices
        slice_indices = np.arange(0, slice_idx_end - slice_idx_start)
        slice_idx_25 = int(np.percentile(slice_indices, 25))
        slice_idx_75 = int(np.percentile(slice_indices, 75))
        percentile_slices = [slice_idx_25, key_slice_idx_offset, slice_idx_75]
        
        # Create visualization
        self.create_visualization(
            img_3D_ori, 
            segs_3D, 
            percentile_slices, 
            base_name,
            key_slice_idx
        )
    
    def create_visualization(self, img_3D_ori, segs_3D, percentile_slices, base_name, key_slice_idx):
        """
        Create the visualization with percentile slices
        
        Args:
            img_3D_ori: Original 3D image
            segs_3D: 3D segmentation mask
            percentile_slices: List of slice indices for visualization
            base_name: Base filename for saving
            key_slice_idx: Key slice index
        """
        fig, axes = plt.subplots(3, 2, figsize=(8, 15))
        
        for ax in axes.flatten():
            ax.axis('off')
        
        row_titles = ['25th percentile image', 'Key slice image', '75th percentile image']
        row_titles_masks = ['25th percentile overlay', 'Key slice overlay', '75th percentile overlay']
        
        for row_idx, slice_idx in enumerate(percentile_slices):
            # Get the 2D slices
            imgs_2D = img_3D_ori[slice_idx].T
            imgs_2D = imgs_2D[:, :, None].repeat(3, axis=-1)
            segs_2D = segs_3D[slice_idx].T
            
            # Show original image
            axes[row_idx, 0].imshow(imgs_2D, cmap='gray')
            axes[row_idx, 0].set_title(row_titles[row_idx], fontsize=14)
            
            # Show image with mask overlay
            axes[row_idx, 1].imshow(imgs_2D, cmap='gray')
            show_mask(segs_2D, ax=axes[row_idx, 1])
            axes[row_idx, 1].set_title(row_titles_masks[row_idx], fontsize=14)
        
        plt.tight_layout()
        
        # Save the figure
        save_path = join(self.vis_save_dir, f'{base_name}_k{key_slice_idx}_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {save_path}")
    
    def visualize_all(self):
        """
        Visualize all cases in the segmentation info CSV
        """
        print(f"Processing {len(self.seg_info)} segmentation results...")
        
        for idx, row in tqdm(self.seg_info.iterrows(), total=len(self.seg_info), 
                            desc="Creating visualizations"):
            try:
                self.visualize_single_case(row)
            except Exception as e:
                print(f"Error processing {row['nii_name']}: {str(e)}")
                continue
    
    def create_comparison_grid(self, max_cases=9):
        """
        Create a grid comparison of multiple cases
        
        Args:
            max_cases: Maximum number of cases to include in the grid
        """
        n_cases = min(len(self.seg_info), max_cases)
        n_cols = 3
        n_rows = (n_cases + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_cases > 1 else [axes]
        
        for idx in range(n_cases):
            row = self.seg_info.iloc[idx]
            ax = axes[idx]
            
            try:
                # Extract information
                seg_filename = row['nii_name']
                key_slice_idx = int(row['key_slice_index'])
                base_name = seg_filename.replace(f'_k{key_slice_idx}_mask.nii.gz', '')
                
                # Load the image and mask
                original_img_path = join(self.pred_save_dir, f'{base_name}_img.nii.gz')
                mask_path = join(self.pred_save_dir, seg_filename)
                
                nii_image = sitk.ReadImage(original_img_path)
                nii_image_data = sitk.GetArrayFromImage(nii_image)
                
                nii_mask = sitk.ReadImage(mask_path)
                segs_3D = sitk.GetArrayFromImage(nii_mask)
                
                # Get windowing parameters
                dicom_windows = row['DICOM_windows']
                lower_bound, upper_bound = dicom_windows.split(',')
                lower_bound, upper_bound = float(lower_bound), float(upper_bound)
                
                # Preprocess the image
                img_3D_ori = self.preprocess_image(nii_image_data, lower_bound, upper_bound)
                
                # Get the key slice (adjust for offset)
                case_name_match = re.findall(r'(\d{6}_\d{2}_\d{2})', base_name)
                range_suffix_match = re.findall(r'\d{3}-\d{3}', base_name)
                
                if case_name_match and range_suffix_match:
                    case_name = case_name_match[0]
                    range_suffix = range_suffix_match[0]
                    slice_range = range_suffix.split('-')
                    slice_idx_start = int(slice_range[0])
                    key_slice_idx_offset = key_slice_idx - slice_idx_start
                    
                    # Get the 2D slices
                    imgs_2D = img_3D_ori[key_slice_idx_offset].T
                    imgs_2D = imgs_2D[:, :, None].repeat(3, axis=-1)
                    segs_2D = segs_3D[key_slice_idx_offset].T
                    
                    # Display
                    ax.imshow(imgs_2D, cmap='gray')
                    show_mask(segs_2D, ax=ax)
                    ax.set_title(f'Case {idx + 1}: {base_name}', fontsize=10)
                    ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                ax.axis('off')
        
        # Turn off remaining axes
        for idx in range(n_cases, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save the grid
        save_path = join(self.vis_save_dir, 'comparison_grid.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison grid: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize MedSAM2 Segmentation Results')
    
    parser.add_argument('--pred_save_dir', type=str, 
                        default='./DeepLesion_results',
                        help='Directory containing prediction results')
    parser.add_argument('--path_DL_info', type=str, 
                        default='CT_DeepLesion/DeepLesion_Dataset_Info.csv',
                        help='Path to CT lesion key slices info CSV')
    parser.add_argument('--seg_info_csv', type=str, 
                        default='./DeepLesion_results/tiny_seg_info202412.csv',
                        help='Path to segmentation info CSV')
    parser.add_argument('--create_grid', action='store_true',
                        help='Create a comparison grid of multiple cases')
    parser.add_argument('--max_cases_grid', type=int, default=9,
                        help='Maximum number of cases to include in comparison grid')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MedSAM2Visualizer(
        pred_save_dir=args.pred_save_dir,
        path_DL_info=args.path_DL_info,
        seg_info_csv=args.seg_info_csv
    )
    
    # Run visualization
    visualizer.visualize_all()
    
    # Create comparison grid if requested
    if args.create_grid:
        visualizer.create_comparison_grid(max_cases=args.max_cases_grid)


if __name__ == '__main__':
    main()
