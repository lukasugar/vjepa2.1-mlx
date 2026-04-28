from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio_v2
import numpy as np
from PIL import Image

from vjepa2_1_mlx.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _resize_and_center_crop(image: Image.Image, image_size: int) -> Image.Image:
    short_side = int(256.0 / 224.0 * image_size)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image.size}")
    scale = short_side / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))


def _normalize(chw_image: np.ndarray) -> np.ndarray:
    mean = np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_DEFAULT_STD, dtype=np.float32)[:, None, None]
    return (chw_image - mean) / std


def preprocess_image(path: str | Path, image_size: int = 384) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = _resize_and_center_crop(image, image_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    chw = _normalize(chw)
    return chw[None, :, None, :, :]


def sample_video_frames(path: str | Path, target_frames: int) -> np.ndarray:
    reader = iio_v2.get_reader(path, format="ffmpeg")
    try:
        frames = np.stack([frame for frame in reader], axis=0)
    finally:
        reader.close()
    if len(frames) == 0:
        raise ValueError(f"video has no frames: {path}")
    indices = np.linspace(0, len(frames) - 1, num=target_frames, dtype=np.int64)
    return frames[indices]


def preprocess_video(path: str | Path, image_size: int = 384, num_frames: int = 64) -> np.ndarray:
    frames = sample_video_frames(path, num_frames)
    processed = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        image = _resize_and_center_crop(image, image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        chw = np.transpose(array, (2, 0, 1))
        processed.append(_normalize(chw))
    stacked = np.stack(processed, axis=1)
    return stacked[None, :, :, :, :]


def load_input_tensor(input_path: str | Path, input_type: str, image_size: int, num_frames: int) -> np.ndarray:
    if input_type == "image":
        return preprocess_image(input_path, image_size=image_size)
    if input_type == "video":
        return preprocess_video(input_path, image_size=image_size, num_frames=num_frames)
    raise ValueError(f"unsupported input_type={input_type}")
