#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Curve:
    label: str
    step: np.ndarray
    value: np.ndarray


def infer_label(path: Path) -> str:
    stem = path.stem
    if "__sac_rgbd__" in stem:
        return "Image-based SAC"
    if "__sac__" in stem:
        return "State-based SAC"
    return path.stem


def load_curve(path: Path) -> Curve:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows, f"CSVが空です: {path}"
    assert reader.fieldnames is not None, f"ヘッダが見つかりません: {path}"
    assert "Step" in reader.fieldnames and "Value" in reader.fieldnames, (
        f"必要な列がありません (Step, Value): {path}"
    )

    step = np.array([float(row["Step"]) for row in rows], dtype=np.float64)
    value = np.array([float(row["Value"]) for row in rows], dtype=np.float64)

    order = np.argsort(step)
    return Curve(label=infer_label(path), step=step[order], value=value[order])


def exponential_smoothing(y: np.ndarray, smoothing: float) -> np.ndarray:
    assert 0.0 <= smoothing <= 1.0, "smoothing は 0.0 以上 1.0 以下で指定してください。"
    if len(y) == 0:
        return y
    if smoothing == 0.0:
        return y

    y_smooth = np.empty_like(y, dtype=np.float64)
    y_smooth[0] = y[0]
    for i in range(1, len(y)):
        y_smooth[i] = y_smooth[i - 1] * smoothing + y[i] * (1.0 - smoothing)
    return y_smooth


def resolve_csv_paths(csv_args: list[Path], default_dir: Path) -> list[Path]:
    if csv_args:
        paths = [p for p in csv_args if p.suffix == ".csv"]
    else:
        paths = sorted(default_dir.glob("*.csv"))

    assert paths, f"CSVが見つかりませんでした: {default_dir}"
    for path in paths:
        assert path.exists(), f"CSVが存在しません: {path}"
    return paths


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RL学習の収束曲線をCSVから描画します。")
    parser.add_argument("csv_files", nargs="*", type=Path, help="入力CSV。未指定なら tests/*.csv")
    parser.add_argument("--output", type=Path, default=Path("tests/rl_convergence.png"))
    parser.add_argument("--smoothing", type=float, default=0.6, help="EMA平滑化係数 (0.0-1.0)")
    parser.add_argument("--max-step", type=float, default=15_000_000.0, help="表示する最大 Step")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    csv_paths = resolve_csv_paths(args.csv_files, default_dir=Path("tests"))
    curves = [load_curve(path) for path in csv_paths]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.0), facecolor="white")
    color_map = {
        "State-based SAC": "#1f77b4",
        "Image-based SAC": "#ff7f0e",
    }
    for curve in curves:
        mask = curve.step <= args.max_step
        if not np.any(mask):
            continue
        x = curve.step[mask]
        y = curve.value[mask]
        y = exponential_smoothing(y, smoothing=args.smoothing)
        ax.plot(
            x / 1_000_000.0,
            y,
            linewidth=1.8,
            label=curve.label,
            color=color_map.get(curve.label),
            solid_capstyle="round",
        )

    ax.set_xlabel("Step Number (million)")
    ax.set_ylabel("Average Dense Reward")
    ax.set_xlim(0, args.max_step / 1_000_000.0)
    ax.grid(True, color="#c7c7c7", linewidth=0.8, alpha=0.45)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#d9d9d9")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
