import os
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd

# 导入 MedSAM2 相关模块
from sam2.build_sam import build_sam2_video_predictor_npz

# --- 1. 配置参数 (在这里修改你的设置) ---
# -----------------------------------------------------

# 模型配置
CHECKPOINT_PATH = "/home/lthpc/Next/MedSAM2/exp_log/MedSAM2_FLARE25_RECIST/checkpoints/checkpoint.pt"
MODEL_CFG_PATH = "configs/sam2.1_hiera_t512.yaml"

# 输入和输出路径
INPUT_IMG_PATH = "/home/lthpc/Datasets/CT_DeepLesion-MedSAM2/datasets--wanglab--CT_DeepLesion-MedSAM2/images/000007_03_01_037-072_0000.nii.gz"
# (可选) 提供基准真相(Ground Truth)掩码以计算Dice分数，如果不需要则设为 None
GT_MASK_PATH = None  # 例如: "/path/to/your/ground_truth_mask.nii.gz"

OUTPUT_DIR = "./inference_results_pro"

# 可视化选项
SAVE_PNG_VISUALS = True  # 是否保存PNG格式的可视化结果

# Prompt 配置
KEY_SLICE_INDEX = 'middle'  # 'middle' 或整数
PROMPT_BBOX = 'center'  # 'center' 或 [x_min, y_min, x_max, y_max]

# CT影像窗宽窗位 (Hounsfield Units)
WINDOW_LOWER_BOUND = -150
WINDOW_UPPER_BOUND = 250

# -----------------------------------------------------
# --- 2. 辅助函数 ---
# -----------------------------------------------------


def getLargestCC(segmentation):
    """获取分割结果中的最大连通域。"""
    labels = measure.label(segmentation, connectivity=3)
    if labels.max() == 0:
        return segmentation
    largestCC_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largestCC = (labels == largestCC_label)
    return largestCC


def calculate_dice_score(pred_mask, gt_mask):
    """计算单个类别的Dice系数。"""
    smooth = 1e-5
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.sum(pred_mask & gt_mask)
    return (2. * intersection + smooth) / (np.sum(pred_mask) + np.sum(gt_mask) + smooth)


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """在图像上显示掩码。"""
    if mask_color is None:
        mask_color = np.array([255, 0, 0])  # 默认为红色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * (mask_color.reshape(1, 1, -1) / 255.0)
    ax.imshow(np.concatenate([mask_image, np.full((h, w, 1), alpha, dtype=float)], axis=2))


def show_box(box, ax, edgecolor='cyan'):
    """在图像上显示边界框。"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """将3D灰度数组的每个切片转换为RGB格式并调整大小。"""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.uint8)
    for i in range(d):
        img_pil = Image.fromarray(array[i])
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size), Image.BILINEAR)
        resized_array[i] = np.array(img_resized).transpose(2, 0, 1)
    return resized_array

# -----------------------------------------------------
# --- 3. 主推理逻辑 ---
# -----------------------------------------------------


def main():
    """执行单张3D影像的推理、评估和可视化。"""
    # 初始化
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    torch.set_float32_matmul_precision('high')

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"模型权重文件未找到: {CHECKPOINT_PATH}")
    if not os.path.exists(INPUT_IMG_PATH):
        raise FileNotFoundError(f"输入影像文件未找到: {INPUT_IMG_PATH}")
    if GT_MASK_PATH and not os.path.exists(GT_MASK_PATH):
        raise FileNotFoundError(f"基准真相掩码文件未找到: {GT_MASK_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出将保存到: {OUTPUT_DIR}")

    # 加载模型
    print("正在加载MedSAM2模型...")
    # 注意: 此处假设OmegaConf解析器已正确配置或脚本不需要它
    predictor = build_sam2_video_predictor_npz(MODEL_CFG_PATH, CHECKPOINT_PATH)
    print("模型加载完成。")

    # 1. 加载和预处理影像
    print(f"正在加载影像: {INPUT_IMG_PATH}")
    sitk_image = sitk.ReadImage(INPUT_IMG_PATH)
    image_data = sitk.GetArrayFromImage(sitk_image)

    image_data_pre = np.clip(image_data, WINDOW_LOWER_BOUND, WINDOW_UPPER_BOUND)
    image_data_pre = (image_data_pre - image_data_pre.min()) / (image_data_pre.max() - image_data_pre.min() + 1e-8) * 255.0
    image_data_pre = np.uint8(image_data_pre)

    # 2. 准备模型输入
    model_input_size = 512
    img_resized = resize_grayscale_to_rgb_and_resize(image_data_pre, model_input_size)
    img_resized = torch.from_numpy(img_resized).float().cuda() / 255.0
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).cuda()
    img_resized = (img_resized - img_mean) / img_std

    # 3. 确定Prompt
    num_slices, height, width = image_data.shape
    key_slice_idx = num_slices // 2 if KEY_SLICE_INDEX == 'middle' else int(KEY_SLICE_INDEX)
    if PROMPT_BBOX == 'center':
        cx, cy = width // 2, height // 2
        box_size = 50
        bbox = np.array([cx - box_size // 2, cy - box_size // 2, cx + box_size // 2, cy + box_size // 2])
    else:
        bbox = np.array(PROMPT_BBOX)
    print(f"关键切片索引: {key_slice_idx}, BBox Prompt: {bbox.tolist()}")

    # 4. 执行推理 (逻辑与之前相同)
    print("开始推理...")
    final_3d_mask = np.zeros_like(image_data, dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, height, width)
        # 正向
        predictor.add_new_points_or_box(inference_state, key_slice_idx, 1, box=bbox)
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            final_3d_mask[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        # 反向
        predictor.reset_state(inference_state)
        predictor.add_new_points_or_box(inference_state, key_slice_idx, 1, box=bbox)
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            final_3d_mask[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
    print("推理完成。")

    # 5. 后处理
    print("正在进行后处理 (获取最大连通域)...")
    if final_3d_mask.max() > 0:
        final_3d_mask = getLargestCC(final_3d_mask)
    final_3d_mask = final_3d_mask.astype(np.uint8)

    # 6. 保存NIfTI分割结果
    output_filename_nii = os.path.basename(INPUT_IMG_PATH).replace(".nii.gz", "_seg.nii.gz")
    output_filepath_nii = os.path.join(OUTPUT_DIR, output_filename_nii)
    print(f"正在保存NIfTI分割结果到: {output_filepath_nii}")
    sitk_mask = sitk.GetImageFromArray(final_3d_mask)
    sitk_mask.CopyInformation(sitk_image)
    sitk.WriteImage(sitk_mask, output_filepath_nii)

    # --- 新增: 7. 指标计算 ---
    dice_score = "N/A"
    if GT_MASK_PATH:
        print(f"正在加载基准真相掩码: {GT_MASK_PATH}")
        sitk_gt_mask = sitk.ReadImage(GT_MASK_PATH)
        gt_mask_data = sitk.GetArrayFromImage(sitk_gt_mask)
        gt_mask_data = (gt_mask_data > 0).astype(np.uint8)  # 确保为二进制

        print("正在计算Dice分数...")
        dice_score = calculate_dice_score(final_3d_mask, gt_mask_data)
        print(f"-> Dice Score: {dice_score:.4f}")

    # --- 新增: 8. 保存PNG可视化结果 ---
    if SAVE_PNG_VISUALS:
        png_output_dir = os.path.join(OUTPUT_DIR, "png_visuals")
        os.makedirs(png_output_dir, exist_ok=True)
        print(f"正在保存PNG可视化图像到: {png_output_dir}")

        for i in range(num_slices):
            # 只保存包含分割结果的切片
            if final_3d_mask[i].max() > 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # 显示原始图像
                ax.imshow(image_data_pre[i], cmap='gray')

                # 叠加分割掩码
                show_mask(final_3d_mask[i], ax, mask_color=np.array([255, 0, 0]))  # 红色掩码

                # 如果是关键切片，则画出prompt
                if i == key_slice_idx:
                    show_box(bbox, ax, edgecolor='cyan')

                ax.axis('off')
                plt.savefig(
                    os.path.join(png_output_dir, f"slice_{i:04d}.png"),
                    bbox_inches='tight', pad_inches=0
                )
                plt.close(fig)

    # --- 新增: 9. 保存结果摘要 ---
    summary = {
        'Input_File': [os.path.basename(INPUT_IMG_PATH)],
        'Key_Slice': [key_slice_idx],
        'Prompt_BBox': [str(bbox.tolist())],
        'Dice_Score': [f"{dice_score:.4f}" if isinstance(dice_score, float) else dice_score],
        'Output_NII': [output_filename_nii]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    summary_df.to_csv(summary_path, mode='a', header=not os.path.exists(summary_path), index=False)
    print(f"结果摘要已保存/追加到: {summary_path}")

    print("全部处理完成！")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 本脚本需要CUDA环境来运行。")
    else:
        main()
