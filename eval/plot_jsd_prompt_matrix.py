from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


COLOR_MAP = {
    "base": "#4C78A8",
    "initdrop2": "#F58518",
    "initdrop2p0p3": "#ECA82C",
    "initdrop8": "#54A24B",
    "initdrop8p0p3": "#E45756",
    "initdrop8p0p3_melloss": "#B279A2",
}

LABEL_MAP = {
    "base": "Base",
    "initdrop2": "InitDrop2",
    "initdrop2p0p3": "InitDrop2p0.3",
    "initdrop8": "InitDrop8",
    "initdrop8p0p3": "InitDrop8p0.3",
    "initdrop8p0p3_melloss": "InitDrop8p0.3+MelLoss",
}

PROMPT_ORDER = {
    "gtprompt8": 0,
    "melprompt8": 1,
    "noprompt0": 2,
}

PROMPT_STYLE = {
    "gtprompt8": {"marker": "o", "face_mode": "filled", "label": "GT Prompt (8 beats)"},
    "melprompt8": {"marker": "^", "face_mode": "white", "label": "Melody-Only Prompt (8 beats)"},
    "noprompt0": {"marker": "s", "face_mode": "white", "label": "No Prompt (from beat 0)"},
}


def parse_run_name(run_dir: str) -> Tuple[str, str]:
    stem = run_dir.replace("generated_samples-", "", 1)
    for suffix in ("gtprompt8", "melprompt8", "noprompt0"):
        end = "-" + suffix
        if stem.endswith(end):
            family = stem[: -len(end)]
            return family, suffix
    raise ValueError(f"Unrecognized prompt-matrix run_dir: {run_dir}")


def load_rows(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            family, mode = parse_run_name(row["run_dir"])
            rows.append(
                {
                    "run_dir": row["run_dir"],
                    "family": family,
                    "family_label": LABEL_MAP.get(family, family),
                    "prompt_mode": mode,
                    "num_pairs": int(float(row["num_pairs"])),
                    "pitch_jsd": float(row["pitch_jsd_mean"]),
                    "onset_jsd": float(row["onset_jsd_mean"]),
                }
            )
    return rows


def plot(rows: List[Dict], out_path: Path, title: str, annotate: bool) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.2), dpi=180)

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)

    for family, items in grouped.items():
        color = COLOR_MAP.get(family, "#777777")
        items = sorted(items, key=lambda x: PROMPT_ORDER.get(x["prompt_mode"], 99))

        if len(items) >= 2:
            xs = [item["onset_jsd"] for item in items]
            ys = [item["pitch_jsd"] for item in items]
            ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.55, zorder=1)

        for item in items:
            style = PROMPT_STYLE[item["prompt_mode"]]
            face = color
            edge = "#111111"
            alpha = 0.95 if item["num_pairs"] >= 19 else 0.55
            size = 100 if item["num_pairs"] >= 19 else 82

            ax.scatter(
                item["onset_jsd"],
                item["pitch_jsd"],
                s=size,
                marker=style["marker"],
                facecolors=face,
                edgecolors=edge,
                linewidths=2.0,
                alpha=alpha,
                zorder=3,
            )

            if annotate:
                pairs_note = "" if item["num_pairs"] >= 19 else f" [n={item['num_pairs']}]"
                ax.annotate(
                    f"{item['family_label']} / {style['label']}{pairs_note}",
                    (item["onset_jsd"], item["pitch_jsd"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.3,
                    color="#222222",
                )

    ax.set_xlabel("Onset JSD (lower is better)")
    ax.set_ylabel("Pitch JSD (lower is better)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    prompt_handles = []
    for mode in ("gtprompt8", "melprompt8", "noprompt0"):
        style = PROMPT_STYLE[mode]
        prompt_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor="#BDBDBD",
                markeredgecolor="#111111",
                markersize=8,
                label=style["label"],
            )
        )
    prompt_legend = ax.legend(handles=prompt_handles, loc="upper left", frameon=True, title="Prompt Type")
    ax.add_artist(prompt_legend)

    model_handles = []
    for family in sorted(grouped.keys(), key=lambda x: LABEL_MAP.get(x, x)):
        model_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLOR_MAP.get(family, "#777777"),
                markeredgecolor="#111111",
                markersize=8,
                label=LABEL_MAP.get(family, family),
            )
        )
    ax.legend(handles=model_handles, loc="upper right", frameon=True, title="Model Type")

    note = (
        "Color = model family; marker shape = prompt condition; "
        "faded points indicate incomplete sample count."
    )
    fig.text(0.02, 0.02, note, fontsize=8.5, color="#444444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prompt-matrix runs in JSD space.")
    parser.add_argument("--csv", required=True, help="by_run csv path")
    parser.add_argument("--out", required=True, help="output path (.png/.svg/.pdf)")
    parser.add_argument("--title", default="Prompt Matrix in JSD Space")
    parser.add_argument("--annotate", action="store_true", help="show text labels next to points")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    plot(rows, Path(args.out), args.title, annotate=args.annotate)


if __name__ == "__main__":
    main()
