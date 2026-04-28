from __future__ import annotations

from pathlib import Path

ARTIFACTS_DIR = Path(".artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"
CONVERTED_DIR = ARTIFACTS_DIR / "converted"
FEATURES_DIR = ARTIFACTS_DIR / "features"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"

REFERENCES_DIR = Path("references")
UPSTREAM_VJEPA2_DIR = REFERENCES_DIR / "vjepa2"
UPSTREAM_VJEPA2_URL = "https://github.com/facebookresearch/vjepa2"
UPSTREAM_VJEPA2_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
SAMPLE_VIDEO_URL = (
    "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/"
    "bowling/-WH-lxmGJVY_000005_000015.mp4"
)
SAMPLE_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
    "coco_sample.png"
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

MODEL_SPECS = {
    "vjepa2_1_vit_base_384": {
        "checkpoint_name": "vjepa2_1_vitb_dist_vitG_384.pt",
        "arch_name": "vit_base",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "img_size": 384,
        "num_frames": 64,
        "patch_size": 16,
        "tubelet_size": 2,
    },
    "vjepa2_1_vit_large_384": {
        "checkpoint_name": "vjepa2_1_vitl_dist_vitG_384.pt",
        "arch_name": "vit_large",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "img_size": 384,
        "num_frames": 64,
        "patch_size": 16,
        "tubelet_size": 2,
    },
}
