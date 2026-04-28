from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent
BLOG_POST = REPO_ROOT / "blog_post_vjepa2_1_mlx.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "blog_post_vjepa2_1_mlx"


def parse_table(markdown: str, header: str) -> list[dict[str, str]]:
    lines = markdown.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != header:
            continue
        separator_idx = idx + 1
        if separator_idx >= len(lines) or not lines[separator_idx].strip().startswith("|"):
            raise ValueError(f"missing separator after table header: {header}")
        rows: list[dict[str, str]] = []
        columns = [part.strip() for part in line.strip().strip("|").split("|")]
        row_idx = idx + 2
        while row_idx < len(lines):
            row = lines[row_idx].strip()
            if not row.startswith("|"):
                break
            values = [part.strip() for part in row.strip("|").split("|")]
            if len(values) != len(columns):
                break
            rows.append(dict(zip(columns, values, strict=True)))
            row_idx += 1
        return rows
    raise ValueError(f"table not found: {header}")


def parse_metric(value: str) -> float | None:
    cleaned = value.strip().strip("`")
    lowered = cleaned.lower()
    if lowered in {"failed", "oom", "/"}:
        return None
    if cleaned.endswith(" ms"):
        cleaned = cleaned[:-3]
    return float(cleaned)


def load_blog_tables(markdown_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    markdown = markdown_path.read_text()
    parity_rows = parse_table(
        markdown,
        "| Model | Input | Cosine similarity | Max abs error | Mean abs error |",
    )
    image_rows = parse_table(
        markdown,
        "| Model | PyTorch CPU | PyTorch MPS | MLX |",
    )
    video_rows = parse_table(
        markdown,
        "| Model | Frames | PyTorch CPU | PyTorch MPS | MLX |",
    )
    return parity_rows, image_rows, video_rows


def make_parity_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [f"{row['Model'].strip('`')} {row['Input']}" for row in rows]
    one_minus_cosine = [max(1.0 - float(row["Cosine similarity"].strip("`")), 1e-16) for row in rows]
    mean_abs_error = [float(row["Mean abs error"].strip("`")) for row in rows]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = sns.color_palette("crest", n_colors=len(rows))

    axes[0].bar(labels, one_minus_cosine, color=palette)
    axes[0].set_yscale("log")
    axes[0].set_title("Parity Gap (1 - cosine similarity)")
    axes[0].set_ylabel("Lower is better")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, mean_abs_error, color=palette)
    axes[1].set_yscale("log")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_ylabel("Lower is better")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("V-JEPA 2.1 MLX Port: Parity Checks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_image_benchmark_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    models = [row["Model"].strip("`") for row in rows]
    backends = ["PyTorch CPU", "PyTorch MPS", "MLX"]
    colors = {"PyTorch CPU": "#7c8697", "PyTorch MPS": "#d08770", "MLX": "#2a9d8f"}

    width = 0.22
    x_positions = list(range(len(models)))
    offsets = {
        "PyTorch CPU": -width,
        "PyTorch MPS": 0.0,
        "MLX": width,
    }

    for backend in backends:
        values = [parse_metric(row[backend]) for row in rows]
        bars = ax.bar(
            [x + offsets[backend] for x in x_positions],
            values,
            width=width,
            label=backend,
            color=colors[backend],
        )
        for bar, value in zip(bars, values, strict=True):
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.02,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Image Inference Latency")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_video_benchmark_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)

    model_order = ["ViT-B/16", "ViT-L/16"]
    backend_order = ["PyTorch CPU", "PyTorch MPS", "MLX"]
    colors = {"PyTorch CPU": "#7c8697", "PyTorch MPS": "#d08770", "MLX": "#2a9d8f"}

    rows_by_model: dict[str, list[dict[str, str]]] = {model: [] for model in model_order}
    for row in rows:
        rows_by_model[row["Model"].strip("`")].append(row)
    for model_rows in rows_by_model.values():
        model_rows.sort(key=lambda item: int(item["Frames"].strip("`")))

    for ax, model in zip(axes, model_order, strict=True):
        model_rows = rows_by_model[model]
        frames = [int(row["Frames"].strip("`")) for row in model_rows]

        for backend in backend_order:
            y_values = [parse_metric(row[backend]) for row in model_rows]
            line_values = [float("nan") if value is None else value for value in y_values]
            ax.plot(
                frames,
                line_values,
                marker="o",
                linewidth=2.5,
                label=backend,
                color=colors[backend],
            )

            for frame, value in zip(frames, y_values, strict=True):
                if value is None:
                    continue
                ax.text(frame, value * 1.05, f"{value:.0f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(model)
        ax.set_xlabel("Frames")
        ax.set_xticks(frames)
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)

    axes[0].set_ylabel("Median latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Video Inference Latency", y=0.98)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-path", type=Path, default=BLOG_POST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    parity_rows, image_rows, video_rows = load_blog_tables(args.markdown_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_parity_figure(parity_rows, args.output_dir / "parity_metrics.png")
    make_image_benchmark_figure(image_rows, args.output_dir / "benchmark_images.png")
    make_video_benchmark_figure(video_rows, args.output_dir / "benchmark_videos.png")

    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
