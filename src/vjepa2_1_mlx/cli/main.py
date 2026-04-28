from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vjepa2_1_mlx.backends.mlx import MLXBackend
from vjepa2_1_mlx.backends.pytorch import PyTorchBackend
from vjepa2_1_mlx.benchmarking import (
    BenchmarkCase,
    render_benchmark_console_summary,
    run_benchmark_suite,
    write_benchmark_artifacts,
)
from vjepa2_1_mlx.compare import compare_features
from vjepa2_1_mlx.constants import MODEL_SPECS, SAMPLE_IMAGE_URL, SAMPLE_VIDEO_URL, SAMPLES_DIR
from vjepa2_1_mlx.preprocessing import load_input_tensor
from vjepa2_1_mlx.utils.checkpoints import download_checkpoint, download_file
from vjepa2_1_mlx.utils.fs import ensure_artifact_dirs

logger = logging.getLogger(__name__)


def backend_from_name(backend: str, model_name: str):
    if backend == "mlx":
        return MLXBackend(model_name)
    if backend == "pytorch":
        return PyTorchBackend(model_name, device="cpu")
    raise ValueError(f"unsupported backend={backend}")


def command_download_assets(args: argparse.Namespace) -> int:
    ensure_artifact_dirs()
    model_names = [args.model] if args.model else list(MODEL_SPECS)
    for model_name in model_names:
        path = download_checkpoint(model_name)
        print(f"checkpoint[{model_name}]={path}")

    image_path = download_file(SAMPLE_IMAGE_URL, SAMPLES_DIR / "sample_image.png")
    video_path = download_file(SAMPLE_VIDEO_URL, SAMPLES_DIR / "sample_video.mp4")
    print(f"sample_image={image_path}")
    print(f"sample_video={video_path}")
    return 0


def _run_single_infer(backend_name: str, model_name: str, input_path: str, input_type: str) -> dict:
    spec = MODEL_SPECS[model_name]
    input_tensor = load_input_tensor(
        input_path=input_path,
        input_type=input_type,
        image_size=spec["img_size"],
        num_frames=spec["num_frames"],
    )
    backend = backend_from_name(backend_name, model_name)
    output = backend.infer(input_tensor)
    return {
        "backend": backend_name,
        "model": model_name,
        "input_type": input_type,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "output_mean": float(output.mean()),
        "output_std": float(output.std()),
    }


def command_infer(args: argparse.Namespace) -> int:
    result = _run_single_infer(args.backend, args.model, args.input_path, args.input_type)
    print(json.dumps(result, indent=2))
    return 0


def command_parity(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[args.model]
    input_tensor = load_input_tensor(
        input_path=args.input_path,
        input_type=args.input_type,
        image_size=spec["img_size"],
        num_frames=spec["num_frames"],
    )
    pytorch_output = PyTorchBackend(args.model, device="cpu").infer(input_tensor)
    mlx_output = MLXBackend(args.model).infer(input_tensor)
    metrics = compare_features(pytorch_output, mlx_output)
    payload = {
        "model": args.model,
        "input_type": args.input_type,
        "input_shape": list(input_tensor.shape),
        "output_shape_pytorch": list(pytorch_output.shape),
        "output_shape_mlx": list(mlx_output.shape),
        **metrics,
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    sample_path = SAMPLES_DIR / ("sample_image.png" if args.input_type == "image" else "sample_video.mp4")
    cases = [BenchmarkCase(model_name=args.model, input_type=args.input_type, input_path=str(sample_path))]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(
        f"Starting benchmark: cases={len(cases)}, warmup_iterations={args.warmup_iterations}, "
        f"timed_iterations={args.timed_iterations}, benchmark_video_num_frames={args.num_frames}"
    )
    results = run_benchmark_suite(
        cases=cases,
        warmup_iterations=args.warmup_iterations,
        timed_iterations=args.timed_iterations,
        benchmark_num_frames=args.num_frames,
    )
    json_path, report_path = write_benchmark_artifacts(results, repo_root=repo_root)
    logger.info("%s", render_benchmark_console_summary(results, json_path=json_path, report_path=report_path))
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "report_path": str(report_path),
                "num_cases": len(cases),
                "num_target_runs": len(results["targets"]),
            },
            indent=2,
        )
    )
    return 0


def _not_implemented(name: str):
    raise NotImplementedError(f"{name} is planned for later milestones")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vjepa2-1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-assets")
    download_parser.add_argument("--model", choices=list(MODEL_SPECS), default=None)
    download_parser.set_defaults(func=command_download_assets)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--backend", choices=["pytorch", "mlx"], required=True)
    infer_parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    infer_parser.add_argument("--input-path", required=True)
    infer_parser.add_argument("--input-type", choices=["image", "video"], required=True)
    infer_parser.set_defaults(func=command_infer)

    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    parity_parser.add_argument("--input-path", required=True)
    parity_parser.add_argument("--input-type", choices=["image", "video"], required=True)
    parity_parser.set_defaults(func=command_parity)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    benchmark_parser.add_argument("--input-type", choices=["image", "video"], required=True)
    benchmark_parser.add_argument("--num-frames", type=int, default=16)
    benchmark_parser.add_argument("--warmup-iterations", type=int, default=2)
    benchmark_parser.add_argument("--timed-iterations", type=int, default=10)
    benchmark_parser.set_defaults(func=command_benchmark)

    for name in ("extract-ssv2-features", "train-ssv2-probe", "eval-ssv2-probe"):
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(func=lambda args, n=name: _not_implemented(n))

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
