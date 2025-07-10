# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch
import numpy as np

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
    NPZSegmentLoader
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

        # logging.info(f"video_names:{self.video_names}")

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        # print(f"video_name in this idx:{video_name}")

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root, sample_rate=self.sample_rate)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.png")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for idx, fpath in enumerate(all_frames[::self.sample_rate]):
            fid = idx  # int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class NPZRawDataset(VOSRawDataset):
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # Read all npz files from folder and its subfolders
        subset = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.endswith('.npz'):
                    # Get the relative path from the root folder
                    rel_path = os.path.relpath(os.path.join(root, file), self.folder)
                    # Remove the .npz extension
                    subset.append(os.path.splitext(rel_path)[0])

        # Read the subset defined in file_list_txt if provided
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f if line.strip() in subset]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")

        # Load NPZ file
        npz_data = np.load(npz_path)

        # Extract frames and masks
        frames = npz_data['imgs'] / 255.0
        # Expand the grayscale images to three channels
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)
        masks = npz_data['gts']

        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]

        # Create VOSFrame objects
        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))

        # Create VOSVideo object
        video = VOSVideo(video_name, idx, vos_frames)

        # Create NPZSegmentLoader
        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])

        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
