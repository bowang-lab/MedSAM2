# MedSAM2

## Encoder

阶段/模块	操作与描述	输出 Shape (B=64)

Input	原始输入图像。	(64, 3, 512, 512)

Hiera (Trunk)	提取4个阶段的多尺度特征。	4个特征图列表

⬇️ FpnNeck 处理	自顶向下处理...	
1. Level 3 (融合)	Hiera Stage 4 (16x16) 特征 -> 1x1卷积 -> d_model=256。	(64, 256, 16, 16)
2. Level 2 (融合)	Hiera Stage 3 (32x32) 特征 -> 1x1卷积 -> 256通道，并与上采样后的 Level 3 特征相加。	(64, 256, 32, 32)
3. Level 1 (仅转换)	Hiera Stage 2 (64x64) 特征 -> 1x1卷积 -> 64通道 (不融合)。	(64, 64, 64, 64)
4. Level 0 (仅转换)	Hiera Stage 1 (128x128) 特征 -> 1x1卷积 -> 32通道 (不融合)。	(64, 32, 128, 128)

⬇️ ImageEncoder 封装		

scalp: 1	丢弃 FpnNeck 输出的最后一层（Level 3, 16x16）。	-

backbone_fpn	[最终输出] 保留的 Level 0, 1, 2 的特征图列表。	list of 3 Tensors (见下)
- [0] - Level 0 特征	(64, 32, 128, 128)
- [1] - Level 1 特征	(64, 64, 64, 64)
- [2] - Level 2 特征	(64, 256, 32, 32)

vision_features	[最终输出] backbone_fpn 列表的最后一个元素。	(64, 256, 32, 32)

vision_pos_enc	[最终输出] 为 backbone_fpn 的每一层生成的位置编码列表，通道数统一为256。	list of 3 Tensors (见下)
- [0] - Level 0 位置编码	(64, 256, 128, 128)
- [1] - Level 1 位置编码	(64, 256, 64, 64)
- [2] - Level 2 位置编码	(64, 256, 32, 32)


Batchsize = train_video_batch_size * num_frames

