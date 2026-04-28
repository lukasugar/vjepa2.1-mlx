from __future__ import annotations

from pathlib import Path

from vjepa2_1_mlx.cli.main import build_parser
from vjepa2_1_mlx.benchmarking import (
    BenchmarkCase,
    render_benchmark_console_summary,
    render_benchmark_report,
    resolve_benchmark_num_frames,
    summarize_latencies_ms,
)


def test_summarize_latencies_ms():
    summary = summarize_latencies_ms([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["median_ms"] == 3.0
    assert summary["p10_ms"] == 1.4
    assert summary["p90_ms"] == 4.6
    assert summary["num_timed_iterations"] == 5


def test_resolve_benchmark_num_frames_uses_override_for_video_only():
    video_case = BenchmarkCase(
        model_name="vjepa2_1_vit_base_384",
        input_type="video",
        input_path="video.mp4",
    )
    image_case = BenchmarkCase(
        model_name="vjepa2_1_vit_base_384",
        input_type="image",
        input_path="image.png",
    )
    assert resolve_benchmark_num_frames(video_case, 16) == 16
    assert resolve_benchmark_num_frames(image_case, 16) == 64


def test_benchmark_parser_requires_model_and_input_type():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--model", "vjepa2_1_vit_base_384", "--input-type", "video"])
    assert args.model == "vjepa2_1_vit_base_384"
    assert args.input_type == "video"
    assert args.num_frames == 16
    assert args.warmup_iterations == 2
    assert args.timed_iterations == 10


def test_render_benchmark_report_includes_failures():
    results = {
        "generated_at": "2026-04-22T12:00:00+00:00",
        "host": {"platform": "Darwin", "processor": "arm", "python": "3.12.9"},
        "software": {"torch": "2.11.0", "mlx": "0.31.2"},
        "settings": {
            "warmup_iterations": 10,
            "timed_iterations": 30,
            "headline_measurement": "forward_only_ms",
            "same_preprocessed_tensor_for_all_targets": True,
            "benchmark_video_num_frames": 16,
            "compile_policy": {
                "pytorch_cpu": "torch.compile(model)",
                "pytorch_mps": "eager model (torch.compile disabled for benchmarking because torch.compile on MPS previously failed with InductorError: AssertionError: (VR[0, 0], VR[0.0, 0.0]))",
                "mlx": "mx.compile(model)",
            },
            "known_compile_failures": {
                "pytorch_mps_torch_compile": {
                    "error_type": "InductorError",
                    "error": "AssertionError: (VR[0, 0], VR[0.0, 0.0])",
                }
            },
        },
        "preprocessing": [
            {
                "model": "vjepa2_1_vit_base_384",
                "input_type": "image",
                "frames": None,
                "output_shape": [1, 3, 1, 384, 384],
                "median_ms": 1.0,
                "p10_ms": 0.8,
                "p90_ms": 1.2,
            }
        ],
        "targets": [
            {
                "model": "vjepa2_1_vit_base_384",
                "input_type": "image",
                "frames": None,
                "target": "pytorch_cpu",
                "status": "ok",
                "median_ms": 2.0,
                "p10_ms": 1.5,
                "p90_ms": 2.5,
                "notes": "",
            },
            {
                "model": "vjepa2_1_vit_base_384",
                "input_type": "image",
                "frames": None,
                "target": "pytorch_mps",
                "status": "ok",
                "median_ms": 1.0,
                "p10_ms": 0.9,
                "p90_ms": 1.1,
                "notes": "Ran without torch.compile because torch.compile on MPS previously failed with InductorError: AssertionError: (VR[0, 0], VR[0.0, 0.0])",
            },
        ],
    }
    report = render_benchmark_report(results, Path(".artifacts/benchmarks/example.json"))
    assert "Milestone 2 Benchmark Results" in report
    assert "`pytorch_cpu`" in report
    assert "| Model | Input | Frames | Target | Status |" in report
    assert "The same preprocessed tensor is reused for all targets within each benchmark case." in report
    assert "Ran without torch.compile because torch.compile on MPS previously failed" in report
    assert "Known compile issue retained for reference" in report


def test_render_benchmark_console_summary_includes_metrics():
    results = {
        "preprocessing": [
            {
                "model": "vjepa2_1_vit_base_384",
                "input_type": "image",
                "frames": None,
                "output_shape": [1, 3, 1, 384, 384],
                "median_ms": 1.0,
                "p10_ms": 0.8,
                "p90_ms": 1.2,
            }
        ],
        "targets": [
            {
                "model": "vjepa2_1_vit_base_384",
                "input_type": "image",
                "frames": None,
                "target": "pytorch_cpu",
                "status": "ok",
                "median_ms": 2.0,
                "p10_ms": 1.5,
                "p90_ms": 2.5,
                "notes": "",
            },
            {
                "model": "vjepa2_1_vit_base_384",
                "input_type": "image",
                "frames": None,
                "target": "pytorch_mps",
                "status": "failed",
                "error_type": "Unavailable",
                "error": "torch.backends.mps.is_available() returned False",
            },
        ],
    }
    summary = render_benchmark_console_summary(
        results,
        Path(".artifacts/benchmarks/example.json"),
        Path("milestone2_benchmark.md"),
    )
    assert "Benchmark summary" in summary
    assert "median=2.000 ms" in summary
    assert "failed with Unavailable" in summary
    assert "shape=[1, 3, 1, 384, 384]" in summary
