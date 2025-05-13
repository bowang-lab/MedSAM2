#!/usr/bin/env python3
"""
MedSAM2 Quick Validation Script
CT Lesion annotation using MedSAM2 with box prompts on key slices

This script performs inference on CT lesion cases using MedSAM2.
Please install MedSAM2 and download the model weights before running.
"""

import os
import re
import argparse
from glob import glob
from collections import OrderedDict
from os.path import join, basename

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure, morphology

from sam2.build_sam import build_sam2_video_predictor_npz


# Set random seeds for reproducibility
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


class MedSAM2Validator:
    """MedSAM2 validation class for CT lesion segmentation"""
    
    def __init__(self, checkpoint, model_cfg, imgs_path, pred_save_dir, path_DL_info):
        """
        Initialize the MedSAM2 validator
        
        Args:
            checkpoint: Path to model checkpoint
            model_cfg: Path to model configuration file
            imgs_path: Path to the directory containing nii.gz files
            pred_save_dir: Directory to save prediction results
            path_DL_info: Path to the CSV file containing CT lesion info
        """
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.imgs_path = imgs_path
        self.pred_save_dir = pred_save_dir
        self.path_DL_info = path_DL_info
        
        # Create output directory if it doesn't exist
        os.makedirs(self.pred_save_dir, exist_ok=True)
        
        # Load dataset info
        self.DL_info = pd.read_csv(self.path_DL_info)
        
        # Initialize predictor
        self.predictor = build_sam2_video_predictor_npz(self.model_cfg, self.checkpoint)
        
        # Image normalization parameters
        self.img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].cuda()
        self.img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].cuda()
    
    @staticmethod
    def get_largest_cc(segmentation):
        """
        Get the largest connected component from a segmentation mask
        
        Args:
            segmentation: Binary segmentation mask
            
        Returns:
            Binary mask containing only the largest connected component
        """
        labels = measure.label(segmentation)
        largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largest_cc
    
    @staticmethod
    def resize_grayscale_to_rgb_and_resize(array, image_size):
        """
        Resize a 3D grayscale NumPy array to an RGB image and then resize it
        
        Args:
            array: Input array of shape (d, h, w)
            image_size: Desired size for the width and height
            
        Returns:
            Resized array of shape (d, 3, image_size, image_size)
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
    
    def prepare_image_tensor(self, img_3D_ori):
        """
        Prepare the image tensor for model input
        
        Args:
            img_3D_ori: Original 3D image
            
        Returns:
            Normalized and resized image tensor
        """
        assert np.max(img_3D_ori) < 256, f'Input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'
        
        img_resized = self.resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        
        # Normalize
        img_resized -= self.img_mean
        img_resized /= self.img_std
        
        return img_resized
    
    def process_single_case(self, nii_fname):
        """
        Process a single CT case
        
        Args:
            nii_fname: Name of the nii.gz file
            
        Returns:
            Dictionary containing segmentation info
        """
        print(f"Processing: {nii_fname}")
        
        # Initialize segmentation info
        seg_info = OrderedDict()
        seg_info['nii_name'] = []
        seg_info['key_slice_index'] = []
        seg_info['DICOM_windows'] = []
        
        # Read the NIfTI image
        nii_image = sitk.ReadImage(join(self.imgs_path, nii_fname))
        nii_image_data = sitk.GetArrayFromImage(nii_image)
        
        # Get corresponding case info
        range_suffix = re.findall(r'\d{3}-\d{3}', nii_fname)[0]
        slice_range = range_suffix.split('-')
        slice_range = [str(int(s)) for s in slice_range]
        slice_range = ', '.join(slice_range)
        
        case_name = re.findall(r'^(\d{6}_\d{2}_\d{2})', nii_fname)[0]
        case_df = self.DL_info[
            self.DL_info['File_name'].str.contains(case_name) & 
            self.DL_info['Slice_range'].str.contains(slice_range)
        ].copy()
        
        if case_df.empty:
            print(f"No matching case info found for {nii_fname}")
            return seg_info
        
        # Process the case
        row = case_df.iloc[0]
        
        # Get windowing parameters
        lower_bound, upper_bound = row['DICOM_windows'].split(',')
        lower_bound, upper_bound = float(lower_bound), float(upper_bound)
        
        # Preprocess the image
        nii_image_data_pre = self.preprocess_image(nii_image_data, lower_bound, upper_bound)
        
        # Get key slice info
        key_slice_idx = int(row['Key_slice_index'])
        slice_range = row['Slice_range']
        slice_idx_start, slice_idx_end = slice_range.split(',')
        slice_idx_start, slice_idx_end = int(slice_idx_start), int(slice_idx_end)
        
        # Get bounding box coordinates
        bbox_coords = row['Bounding_boxes'].split(',')
        bbox_coords = [int(float(coord)) for coord in bbox_coords]
        bbox = np.array([bbox_coords[1], bbox_coords[0], bbox_coords[3], bbox_coords[2]])  # y_min, x_min, y_max, x_max
        
        # Calculate key slice offset
        key_slice_idx_offset = key_slice_idx - slice_idx_start
        key_slice_img = nii_image_data_pre[key_slice_idx_offset, :, :]
        
        # Get video dimensions
        video_height = key_slice_img.shape[0]
        video_width = key_slice_img.shape[1]
        
        # Prepare image tensor
        img_resized = self.prepare_image_tensor(nii_image_data_pre)
        
        # Perform segmentation
        segs_3D = self.segment_volume(
            img_resized, 
            bbox, 
            key_slice_idx_offset, 
            video_height, 
            video_width,
            nii_image_data_pre.shape
        )
        
        # Post-process and save results
        if np.max(segs_3D) > 0:
            segs_3D = self.get_largest_cc(segs_3D)
            segs_3D = np.uint8(segs_3D)
            
            # Save results
            self.save_results(
                nii_image_data_pre, 
                segs_3D, 
                nii_image, 
                nii_fname, 
                key_slice_idx
            )
            
            seg_info['nii_name'].append(f"{nii_fname.split('.nii.gz')[0]}_k{key_slice_idx}_mask.nii.gz")
            seg_info['key_slice_index'].append(key_slice_idx)
            seg_info['DICOM_windows'].append(row['DICOM_windows'])
        
        return seg_info
    
    def segment_volume(self, img_resized, bbox, key_slice_idx_offset, video_height, video_width, volume_shape):
        """
        Perform 3D segmentation using SAM2
        
        Args:
            img_resized: Resized and normalized image tensor
            bbox: Bounding box coordinates
            key_slice_idx_offset: Key slice index offset
            video_height: Video frame height
            video_width: Video frame width
            volume_shape: Shape of the output volume
            
        Returns:
            3D segmentation mask
        """
        segs_3D = np.zeros(volume_shape, dtype=np.uint8)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Initialize inference state
            inference_state = self.predictor.init_state(img_resized, video_height, video_width)
            
            # Add bounding box prompt
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice_idx_offset,
                obj_id=1,
                box=bbox,
            )
            
            # Propagate forward
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            
            # Reset state for backward propagation
            self.predictor.reset_state(inference_state)
            
            # Add bounding box prompt again
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice_idx_offset,
                obj_id=1,
                box=bbox,
            )
            
            # Propagate backward
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            
            self.predictor.reset_state(inference_state)
        
        return segs_3D
    
    def save_results(self, img_3D_ori, segs_3D, nii_image, nii_fname, key_slice_idx):
        """
        Save the segmentation results
        
        Args:
            img_3D_ori: Original image data
            segs_3D: Segmentation mask
            nii_image: Original NIfTI image
            nii_fname: NIfTI filename
            key_slice_idx: Key slice index
        """
        # Create SITK images
        sitk_image = sitk.GetImageFromArray(img_3D_ori)
        sitk_image.CopyInformation(nii_image)
        
        sitk_mask = sitk.GetImageFromArray(segs_3D)
        sitk_mask.CopyInformation(nii_image)
        
        # Save image and mask
        img_save_path = os.path.join(self.pred_save_dir, nii_fname.replace('.nii.gz', '_img.nii.gz'))
        mask_save_path = os.path.join(
            self.pred_save_dir, 
            f"{nii_fname.split('.nii.gz')[0]}_k{key_slice_idx}_mask.nii.gz"
        )
        
        sitk.WriteImage(sitk_image, img_save_path)
        sitk.WriteImage(sitk_mask, mask_save_path)
        
        print(f"Saved: {mask_save_path}")
    
    def run_validation(self, process_all=False):
        """
        Run validation on the dataset
        
        Args:
            process_all: If True, process all files; if False, process only the first file
        """
        # Get list of nii.gz files
        nii_files = sorted([f for f in os.listdir(self.imgs_path) if f.endswith('.nii.gz')])
        
        if not process_all:
            nii_files = nii_files[:1]  # Process only the first file for demo
        
        all_seg_info = OrderedDict()
        all_seg_info['nii_name'] = []
        all_seg_info['key_slice_index'] = []
        all_seg_info['DICOM_windows'] = []
        
        # Process each file
        for nii_fname in tqdm(nii_files, desc="Processing files"):
            seg_info = self.process_single_case(nii_fname)
            
            # Aggregate results
            for key in seg_info:
                all_seg_info[key].extend(seg_info[key])
        
        # Save aggregated segmentation info to CSV
        if all_seg_info['nii_name']:
            seg_info_df = pd.DataFrame(all_seg_info)
            csv_path = join(self.pred_save_dir, 'tiny_seg_info202412.csv')
            seg_info_df.to_csv(csv_path, index=False)
            print(f"Saved segmentation info to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='MedSAM2 Quick Validation for CT Lesion Segmentation')
    
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/MedSAM2_latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--model_cfg', type=str, 
                        default='configs/sam2.1_hiera_t512.yaml',
                        help='Path to model configuration')
    parser.add_argument('--imgs_path', type=str, 
                        default='./data',
                        help='Path to directory containing nii.gz files')
    parser.add_argument('--pred_save_dir', type=str, 
                        default='./DeepLesion_results',
                        help='Directory to save prediction results')
    parser.add_argument('--path_DL_info', type=str, 
                        default='CT_DeepLesion/DeepLesion_Dataset_Info.csv',
                        help='Path to CT lesion key slices info CSV')
    parser.add_argument('--process_all', action='store_true',
                        help='Process all files instead of just the first one')
    
    args = parser.parse_args()
    
    # Create validator and run validation
    validator = MedSAM2Validator(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        imgs_path=args.imgs_path,
        pred_save_dir=args.pred_save_dir,
        path_DL_info=args.path_DL_info
    )
    
    validator.run_validation(process_all=args.process_all)


if __name__ == '__main__':
    main()
