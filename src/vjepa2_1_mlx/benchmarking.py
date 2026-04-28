from __future__ import annotations

import gc
import json
import logging
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from vjepa2_1_mlx.backends.mlx import MLXBackend
from vjepa2_1_mlx.backends.pytorch import PyTorchBackend
from vjepa2_1_mlx.constants import BENCHMARKS_DIR, MODEL_SPECS
from vjepa2_1_mlx.preprocessing import load_input_tensor
from vjepa2_1_mlx.utils.fs import ensure_artifact_dirs

logger = logging.getLogger(__name__)

MPS_COMPILE_ERROR_TYPE = "InductorError"
MPS_COMPILE_ERROR = (
    "AssertionError: (VR[0, 0], VR[0.0, 0.0])\n\n"
    "Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace "
    "(please do this especially if you're reporting a bug to PyTorch). "
    'For even more developer context, set TORCH_LOGS="+dynamo"'
)


@dataclass(frozen=True)
class BenchmarkCase:
    model_name: str
    input_type: str
    input_path: str


def resolve_benchmark_num_frames(case: BenchmarkCase, benchmark_num_frames: int | None) -> int:
    spec = MODEL_SPECS[case.model_name]
    if case.input_type == "video" and benchmark_num_frames is not None:
        return benchmark_num_frames
    return spec["num_frames"]


def frame_count_for_result(case: BenchmarkCase, benchmark_num_frames: int | None) -> int | None:
    if case.input_type != "video":
        return None
    return resolve_benchmark_num_frames(case, benchmark_num_frames)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    weight = position - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def summarize_latencies_ms(latencies_ms: list[float]) -> dict[str, float]:
    values = sorted(latencies_ms)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": _quantile(values, 0.10),
        "p90_ms": _quantile(values, 0.90),
        "min_ms": values[0],
        "max_ms": values[-1],
        "num_timed_iterations": len(values),
    }


def _cleanup_backend_resources() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    mx.clear_cache()


def benchmark_preprocessing(
    case: BenchmarkCase,
    warmup_iterations: int,
    timed_iterations: int,
    benchmark_num_frames: int | None = None,
) -> dict[str, Any]:
    spec = MODEL_SPECS[case.model_name]
    num_frames = resolve_benchmark_num_frames(case, benchmark_num_frames)

    for _ in range(warmup_iterations):
        _ = load_input_tensor(case.input_path, case.input_type, spec["img_size"], num_frames)

    latencies_ms: list[float] = []
    last_shape: list[int] | None = None
    for _ in range(timed_iterations):
        start = time.perf_counter()
        tensor = load_input_tensor(case.input_path, case.input_type, spec["img_size"], num_frames)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        last_shape = list(tensor.shape)

    return {
        "model": case.model_name,
        "input_type": case.input_type,
        "frames": frame_count_for_result(case, benchmark_num_frames),
        "input_path": case.input_path,
        "output_shape": last_shape,
        "latencies_ms": latencies_ms,
        **summarize_latencies_ms(latencies_ms),
    }


def _benchmark_pytorch_target(
    case: BenchmarkCase,
    device: str,
    input_tensor: np.ndarray,
    warmup_iterations: int,
    timed_iterations: int,
    use_compile: bool,
    benchmark_num_frames: int | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    backend: PyTorchBackend | None = None
    try:
        backend = PyTorchBackend(case.model_name, device=device)
        model = torch.compile(backend.model) if use_compile else backend.model
        tensor = torch.from_numpy(input_tensor).to(device, dtype=torch.float32)

        with torch.inference_mode():
            for _ in range(warmup_iterations):
                output = model(tensor)
            if device == "mps":
                torch.mps.synchronize()

            latencies_ms: list[float] = []
            for _ in range(timed_iterations):
                start = time.perf_counter()
                output = model(tensor)
                if device == "mps":
                    torch.mps.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(elapsed_ms)

        return {
            "target": f"pytorch_{device}",
            "backend": "pytorch",
            "device": device,
            "compiled": use_compile,
            "status": "ok",
            "model": case.model_name,
            "input_type": case.input_type,
            "frames": frame_count_for_result(case, benchmark_num_frames),
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape),
            "latencies_ms": latencies_ms,
            "notes": notes,
            **summarize_latencies_ms(latencies_ms),
        }
    except Exception as exc:
        return {
            "target": f"pytorch_{device}",
            "backend": "pytorch",
            "device": device,
            "compiled": use_compile,
            "status": "failed",
            "model": case.model_name,
            "input_type": case.input_type,
            "frames": frame_count_for_result(case, benchmark_num_frames),
            "input_shape": list(input_tensor.shape),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "notes": notes,
        }
    finally:
        del backend
        _cleanup_backend_resources()


def _benchmark_mlx_target(
    case: BenchmarkCase,
    input_tensor: np.ndarray,
    warmup_iterations: int,
    timed_iterations: int,
    benchmark_num_frames: int | None = None,
) -> dict[str, Any]:
    backend: MLXBackend | None = None
    try:
        backend = MLXBackend(case.model_name)
        compiled_model = mx.compile(backend.model)
        tensor = mx.array(input_tensor)

        for _ in range(warmup_iterations):
            output = compiled_model(tensor)
            mx.eval(output)

        latencies_ms: list[float] = []
        for _ in range(timed_iterations):
            start = time.perf_counter()
            output = compiled_model(tensor)
            mx.eval(output)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

        return {
            "target": "mlx",
            "backend": "mlx",
            "device": "apple_silicon",
            "compiled": True,
            "status": "ok",
            "model": case.model_name,
            "input_type": case.input_type,
            "frames": frame_count_for_result(case, benchmark_num_frames),
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape),
            "latencies_ms": latencies_ms,
            **summarize_latencies_ms(latencies_ms),
        }
    except Exception as exc:
        return {
            "target": "mlx",
            "backend": "mlx",
            "device": "apple_silicon",
            "compiled": True,
            "status": "failed",
            "model": case.model_name,
            "input_type": case.input_type,
            "frames": frame_count_for_result(case, benchmark_num_frames),
            "input_shape": list(input_tensor.shape),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        del backend
        _cleanup_backend_resources()


def run_benchmark_suite(
    cases: list[BenchmarkCase],
    warmup_iterations: int,
    timed_iterations: int,
    benchmark_num_frames: int | None = None,
) -> dict[str, Any]:
    ensure_artifact_dirs()
    started_at = datetime.now().astimezone()

    preprocessing_results = []
    target_results = []
    for case in cases:
        spec = MODEL_SPECS[case.model_name]
        num_frames = resolve_benchmark_num_frames(case, benchmark_num_frames)
        logger.info("Testing %s: model=%s", case.input_type, case.model_name)
        logger.info("Running shared preprocessing benchmark")
        preprocessing_results.append(
            benchmark_preprocessing(
                case,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                benchmark_num_frames=benchmark_num_frames,
            )
        )
        shared_input_tensor = load_input_tensor(case.input_path, case.input_type, spec["img_size"], num_frames)
        logger.info("Running PyTorch CPU backend with torch.compile")
        target_results.append(
            _benchmark_pytorch_target(
                case,
                device="cpu",
                input_tensor=shared_input_tensor,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                use_compile=True,
                benchmark_num_frames=benchmark_num_frames,
            )
        )
        if torch.backends.mps.is_available():
            # Keep MPS on eager execution for the active benchmark path. In this repo,
            # torch.compile on MPS currently trips TorchInductor in the upstream RoPE path
            # with the recorded AssertionError below, so we report that as known context
            # instead of folding a compile failure into the main latency table.
            logger.info("Running PyTorch MPS backend without torch.compile")
            target_results.append(
                _benchmark_pytorch_target(
                    case,
                    device="mps",
                    input_tensor=shared_input_tensor,
                    warmup_iterations=warmup_iterations,
                    timed_iterations=timed_iterations,
                    use_compile=False,
                    benchmark_num_frames=benchmark_num_frames,
                    notes=(
                        "Ran without torch.compile because torch.compile on MPS previously failed "
                        f"with {MPS_COMPILE_ERROR_TYPE}: {MPS_COMPILE_ERROR.splitlines()[0]}"
                    ),
                )
            )
        else:
            logger.info("Skipping PyTorch MPS backend because MPS is unavailable")
            target_results.append(
                {
                    "target": "pytorch_mps",
                    "backend": "pytorch",
                    "device": "mps",
                    "compiled": False,
                    "status": "failed",
                    "model": case.model_name,
                    "input_type": case.input_type,
                    "frames": frame_count_for_result(case, benchmark_num_frames),
                    "error_type": "Unavailable",
                    "error": "torch.backends.mps.is_available() returned False",
                }
            )
        logger.info("Running MLX backend with mx.compile")
        target_results.append(
            _benchmark_mlx_target(
                case,
                input_tensor=shared_input_tensor,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                benchmark_num_frames=benchmark_num_frames,
            )
        )

    return {
        "generated_at": started_at.isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "software": {
            "torch": torch.__version__,
            "mlx": mx.__version__,
        },
        "settings": {
            "warmup_iterations": warmup_iterations,
            "timed_iterations": timed_iterations,
            "headline_measurement": "forward_only_ms",
            "same_preprocessed_tensor_for_all_targets": True,
            "benchmark_video_num_frames": benchmark_num_frames,
            "compile_policy": {
                "pytorch_cpu": "torch.compile(model)",
                "pytorch_mps": (
                    "eager model (torch.compile disabled for benchmarking because torch.compile on MPS "
                    f"previously failed with {MPS_COMPILE_ERROR_TYPE}: {MPS_COMPILE_ERROR.splitlines()[0]})"
                ),
                "mlx": "mx.compile(model)",
            },
            "known_compile_failures": {
                "pytorch_mps_torch_compile": {
                    "error_type": MPS_COMPILE_ERROR_TYPE,
                    "error": MPS_COMPILE_ERROR,
                }
            },
        },
        "preprocessing": preprocessing_results,
        "targets": target_results,
    }


def write_benchmark_artifacts(results: dict[str, Any], repo_root: Path) -> tuple[Path, Path]:
    ensure_artifact_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = repo_root / BENCHMARKS_DIR / f"benchmark_{timestamp}.json"
    latest_json_path = repo_root / BENCHMARKS_DIR / "latest_benchmark.json"
    report_path = repo_root / BENCHMARKS_DIR / f"benchmark_{timestamp}.md"
    latest_report_path = repo_root / BENCHMARKS_DIR / "latest_benchmark.md"

    json_payload = json.dumps(results, indent=2)
    json_path.write_text(json_payload)
    latest_json_path.write_text(json_payload)
    report_payload = render_benchmark_report(results, json_path)
    report_path.write_text(report_payload)
    latest_report_path.write_text(report_payload)
    return json_path, report_path


def render_benchmark_console_summary(
    results: dict[str, Any],
    json_path: Path,
    report_path: Path,
) -> str:
    lines = [
        "Benchmark summary",
        f"JSON: {json_path}",
        f"Report: {report_path}",
        "",
        "Forward-only results",
    ]

    for result in results["targets"]:
        frames = result.get("frames")
        frame_label = f" frames={frames}" if frames is not None else ""
        prefix = f"- {result['model']} {result['input_type']} {result['target']}:"
        if result["status"] == "ok":
            note = result.get("notes") or ""
            lines.append(
                f"{prefix}{frame_label} median={result['median_ms']:.3f} ms, "
                f"p10={result['p10_ms']:.3f} ms, p90={result['p90_ms']:.3f} ms"
                + (f" [{note}]" if note else "")
            )
        else:
            lines.append(
                f"{prefix}{frame_label} failed with {result.get('error_type', 'Error')}: {result.get('error', '')}"
            )

    lines.extend(["", "Preprocessing results"])
    for result in results["preprocessing"]:
        frame_label = f", frames={result['frames']}" if result.get("frames") is not None else ""
        lines.append(
            f"- {result['model']} {result['input_type']}: "
            f"median={result['median_ms']:.3f} ms, "
            f"p10={result['p10_ms']:.3f} ms, "
            f"p90={result['p90_ms']:.3f} ms, "
            f"shape={result['output_shape']}{frame_label}"
        )

    return "\n".join(lines)


def _format_success_row(result: dict[str, Any]) -> str:
    notes = result.get("notes") or ""
    frames = result.get("frames")
    frame_value = f"`{frames}`" if frames is not None else "`-`"
    return (
        f"| `{result['model']}` | `{result['input_type']}` | {frame_value} | `{result['target']}` | ok | "
        f"`{result['median_ms']:.3f}` | `{result['p10_ms']:.3f}` | `{result['p90_ms']:.3f}` | {notes} |"
    )


def _format_failure_row(result: dict[str, Any]) -> str:
    message = result.get("error", "").replace("\n", " ")
    if len(message) > 120:
        message = message[:117] + "..."
    frames = result.get("frames")
    frame_value = f"`{frames}`" if frames is not None else "`-`"
    return (
        f"| `{result['model']}` | `{result['input_type']}` | {frame_value} | `{result['target']}` | failed | "
        f"`-` | `-` | `-` | {message} |"
    )


def render_benchmark_report(results: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# Milestone 2 Benchmark Results",
        "",
        "These results were produced locally on April 22, 2026 using the milestone-2 benchmark harness.",
        "",
        "## Methodology",
        "",
        f"- Warmup iterations per case: `{results['settings']['warmup_iterations']}`",
        f"- Timed iterations per case: `{results['settings']['timed_iterations']}`",
        "- Headline numbers measure model forward pass only after preprocessing is complete.",
        "- The same preprocessed tensor is reused for all targets within each benchmark case.",
        "- Preprocessing timing is reported separately.",
        f"- Benchmark video frame count override: `{results['settings']['benchmark_video_num_frames']}`",
        "- Compilation settings:",
        f"  - PyTorch CPU: `{results['settings']['compile_policy']['pytorch_cpu']}`",
        f"  - PyTorch MPS: `{results['settings']['compile_policy']['pytorch_mps']}`",
        f"  - MLX: `{results['settings']['compile_policy']['mlx']}`",
        "- Known compile issue retained for reference:",
        "  - PyTorch MPS `torch.compile`: "
        f"`{results['settings']['known_compile_failures']['pytorch_mps_torch_compile']['error_type']}`: "
        f"{results['settings']['known_compile_failures']['pytorch_mps_torch_compile']['error'].splitlines()[0]}",
        "",
        "## Environment",
        "",
        f"- Generated at: `{results['generated_at']}`",
        f"- Platform: `{results['host']['platform']}`",
        f"- Processor: `{results['host']['processor']}`",
        f"- Python: `{results['host']['python']}`",
        f"- Torch: `{results['software']['torch']}`",
        f"- MLX: `{results['software']['mlx']}`",
        f"- Raw benchmark JSON: [{json_path.name}]({json_path.as_posix()})",
        "",
        "## Forward-Only Results",
        "",
        "| Model | Input | Frames | Target | Status | Median (ms) | P10 (ms) | P90 (ms) | Notes |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]

    for result in results["targets"]:
        if result["status"] == "ok":
            lines.append(_format_success_row(result)[:-1] + " |")
        else:
            lines.append(_format_failure_row(result))

    lines.extend(
        [
            "",
            "## Preprocessing Results",
            "",
            "| Model | Input | Frames | Output shape | Median (ms) | P10 (ms) | P90 (ms) |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for result in results["preprocessing"]:
        frame_value = f"`{result['frames']}`" if result.get("frames") is not None else "`-`"
        lines.append(
            f"| `{result['model']}` | `{result['input_type']}` | {frame_value} | `{result['output_shape']}` | "
            f"`{result['median_ms']:.3f}` | `{result['p10_ms']:.3f}` | `{result['p90_ms']:.3f}` |"
        )

    failures = [result for result in results["targets"] if result["status"] != "ok"]
    if failures:
        lines.extend(["", "## Failed Runs", ""])
        for result in failures:
            lines.append(
                f"- `{result['target']}` on `{result['model']}` / `{result['input_type']}` failed with "
                f"`{result.get('error_type', 'Error')}`: {result.get('error', '')}"
            )

    return "\n".join(lines) + "\n"
