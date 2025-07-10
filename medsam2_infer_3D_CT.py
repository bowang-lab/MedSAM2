from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse

from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz
import SimpleITK as sitk
from skimage import measure, morphology

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="/home/lthpc/Next/MedSAM2/exp_log/MedSAM2_FLARE25_RECIST/checkpoints/checkpoint.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="configs/sam2.1_hiera_t512.yaml",
    help='model config',
)

parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default="/home/lthpc/Next/MedSAM2/data/validation_public_npz",
    help='imgs path',
)
parser.add_argument(
    '--gts_path',
    default=None,
    help='simulate prompts based on ground truth',
)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./DeeLesion_results",
    help='path to save segmentation results',
)
# add option to propagate with either box or mask
parser.add_argument(
    '--propagate_with_box',
    default=True,
    action='store_true',
    help='whether to propagate with box'
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
imgs_path = args.imgs_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
os.makedirs(pred_save_dir, exist_ok=True)
propagate_with_box = args.propagate_with_box


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
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
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.

    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.

    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array

    return resized_array


def mask2D_to_bbox(gt2D, max_shift=20):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes


def mask3D_to_bbox(gt3D, max_shift=20):
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d


# initialized predictor
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
print(f"load model succ")
DL_info = pd.read_csv('CT_DeepLesion/DeepLesion_Dataset_Info.csv')
nii_fnames = sorted(os.listdir(imgs_path))
nii_fnames = [i for i in nii_fnames if i.endswith('.nii.gz')]
nii_fnames = [i for i in nii_fnames if not i.startswith('._')]
print(f'Processing {len(nii_fnames)} nii files')
seg_info = OrderedDict()
seg_info['nii_name'] = []
seg_info['key_slice_index'] = []
seg_info['DICOM_windows'] = []


for nii_fname in tqdm(nii_fnames):
    # get corresponding case info
    range_suffix = re.findall(r'\d{3}-\d{3}', nii_fname)[0]
    slice_range = range_suffix.split('-')
    slice_range = [str(int(s)) for s in slice_range]
    slice_range = ', '.join(slice_range)
    nii_image = sitk.ReadImage(join(imgs_path, nii_fname))
    nii_image_data = sitk.GetArrayFromImage(nii_image)

    case_name = re.findall(r'^(\d{6}_\d{2}_\d{2})', nii_fname)[0]
    case_df = DL_info[
        DL_info['File_name'].str.contains(case_name) &
        DL_info['Slice_range'].str.contains(slice_range)
    ].copy()

    segs_3D = np.zeros(nii_image_data.shape, dtype=np.uint8)

    for row_id, row in case_df.iterrows():
        # print(f'Processing {case_name} tumor {tumor_idx}')
        # get the key slice info
        lower_bound, upper_bound = row['DICOM_windows'].split(',')
        lower_bound, upper_bound = float(lower_bound), float(upper_bound)
        nii_image_data_pre = np.clip(nii_image_data, lower_bound, upper_bound)
        nii_image_data_pre = (nii_image_data_pre - np.min(nii_image_data_pre))/(np.max(nii_image_data_pre)-np.min(nii_image_data_pre))*255.0
        nii_image_data_pre = np.uint8(nii_image_data_pre)
        key_slice_idx = row['Key_slice_index']
        key_slice_idx = int(key_slice_idx)
        slice_range = row['Slice_range']
        slice_idx_start, slice_idx_end = slice_range.split(',')
        slice_idx_start, slice_idx_end = int(slice_idx_start), int(slice_idx_end)
        bbox_coords = row['Bounding_boxes']
        bbox_coords = bbox_coords.split(',')
        bbox_coords = [int(float(coord)) for coord in bbox_coords]
        # bbox_coords = expand_box(bbox_coords)
        bbox = np.array(bbox_coords)  # y_min, x_min, y_max, x_max
        bbox = np.array([bbox[1], bbox[0], bbox[3], bbox[2]])

        key_slice_idx_offset = key_slice_idx - slice_idx_start
        key_slice_img = nii_image_data_pre[key_slice_idx_offset, :, :]

        img_3D_ori = nii_image_data_pre
        assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'

        video_height = key_slice_img.shape[0]
        video_width = key_slice_img.shape[1]
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        img_mean = (0.485, 0.456, 0.406)
        img_std = (0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
        img_resized -= img_mean
        img_resized /= img_std
        z_mids = []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            if propagate_with_box:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_slice_idx_offset,
                    obj_id=1,
                    box=bbox,
                )
            else:  # gt
                pass

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            predictor.reset_state(inference_state)
            if propagate_with_box:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_slice_idx_offset,
                    obj_id=1,
                    box=bbox,
                )
            else:  # gt
                pass

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            predictor.reset_state(inference_state)
        if np.max(segs_3D) > 0:
            segs_3D = getLargestCC(segs_3D)
            segs_3D = np.uint8(segs_3D)
        sitk_image = sitk.GetImageFromArray(img_3D_ori)
        sitk_image.CopyInformation(nii_image)
        sitk_mask = sitk.GetImageFromArray(segs_3D)
        sitk_mask.CopyInformation(nii_image)
        # save single lesion
        key_slice_idx = row['Key_slice_index']
        save_seg_name = nii_fname.split('.nii.gz')[0] + f'_k{key_slice_idx}_mask.nii.gz'
        sitk.WriteImage(sitk_image, os.path.join(pred_save_dir, nii_fname.replace('.nii.gz', '_img.nii.gz')))
        sitk.WriteImage(sitk_mask, os.path.join(pred_save_dir, save_seg_name))
        seg_info['nii_name'].append(save_seg_name)
        seg_info['key_slice_index'].append(key_slice_idx)
        seg_info['DICOM_windows'].append(row['DICOM_windows'])

    seg_info_df = pd.DataFrame(seg_info)
    seg_info_df.to_csv(join(pred_save_dir, 'tiny_seg_info202412.csv'), index=False)
