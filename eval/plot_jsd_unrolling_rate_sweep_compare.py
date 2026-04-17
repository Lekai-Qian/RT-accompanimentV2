from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


COLOR_MAP = {
    "base": "#4C78A8",
    "initdrop8p0p3": "#E45756",
    "initdrop8p0p3_melloss": "#B279A2",
}

LABEL_MAP = {
    "base": "Base",
    "initdrop8p0p3": "InitDrop8p0.3",
    "initdrop8p0p3_melloss": "InitDrop8p0.3+MelLoss",
}

SETTING_ORDER = {
    "i1_g1_p100": 0,
    "i1_g3_p080": 1,
    "i1_g3_p050": 2,
    "i1_g3_p020": 3,
    "i1-g1-p100": 0,
    "i1-g3-p080": 1,
    "i1-g3-p050": 2,
    "i1-g3-p020": 3,
}

SETTING_STYLE = {
    "i1_g1_p100": {"marker": "o", "label": "I=1, G=1, p=1.0"},
    "i1_g3_p080": {"marker": "s", "label": "I=1, G=3, p=0.8"},
    "i1_g3_p050": {"marker": "^", "label": "I=1, G=3, p=0.5"},
    "i1_g3_p020": {"marker": "D", "label": "I=1, G=3, p=0.2"},
    "i1-g1-p100": {"marker": "o", "label": "I=1, G=1, p=1.0"},
    "i1-g3-p080": {"marker": "s", "label": "I=1, G=3, p=0.8"},
    "i1-g3-p050": {"marker": "^", "label": "I=1, G=3, p=0.5"},
    "i1-g3-p020": {"marker": "D", "label": "I=1, G=3, p=0.2"},
}


def parse_run_name(run_dir: str) -> Tuple[str, str]:
    stem = run_dir.replace("generated_samples-unrolling-", "", 1)
    for suffix in (
        "i1_g1_p100",
        "i1_g3_p080",
        "i1_g3_p050",
        "i1_g3_p020",
        "i1-g1-p100",
        "i1-g3-p080",
        "i1-g3-p050",
        "i1-g3-p020",
    ):
        token = f"-prompt8-{suffix}"
        if token in stem:
            family = stem.split(token)[0]
            return family, suffix
    raise ValueError(f"Unrecognized rate-sweep run_dir: {run_dir}")


def load_rows(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            family, setting = parse_run_name(row["run_dir"])
            actual_pairs = int(float(row["num_pairs"]))
            expected_pairs = 19
            effective_pairs = min(actual_pairs, expected_pairs)
            if actual_pairs < expected_pairs:
                status = "incomplete"
            elif actual_pairs > expected_pairs:
                status = "overflow"
            else:
                status = "complete"
            rows.append(
                {
                    "run_dir": row["run_dir"],
                    "family": family,
                    "family_label": LABEL_MAP.get(family, family),
                    "setting": setting,
                    "num_pairs": actual_pairs,
                    "effective_pairs": effective_pairs,
                    "completion_rate": effective_pairs / expected_pairs,
                    "status": status,
                    "pitch_jsd": float(row["pitch_jsd_mean"]),
                    "onset_jsd": float(row["onset_jsd_mean"]),
                }
            )
    return rows


def plot(rows: List[Dict], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=180)

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)

    for family, items in grouped.items():
        color = COLOR_MAP.get(family, "#777777")
        items = sorted(items, key=lambda x: SETTING_ORDER.get(x["setting"], 99))

        if len(items) >= 2:
            xs = [item["onset_jsd"] for item in items]
            ys = [item["pitch_jsd"] for item in items]
            ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.55, zorder=1)

        for item in items:
            style = SETTING_STYLE[item["setting"]]
            edgecolor = "#111111"
            alpha = 0.95
            if item["status"] == "incomplete":
                edgecolor = "#C62828"
                alpha = 0.55
            elif item["status"] == "overflow":
                edgecolor = "#6A1B9A"
                alpha = 0.75
            ax.scatter(
                item["onset_jsd"],
                item["pitch_jsd"],
                s=105,
                marker=style["marker"],
                facecolors=color,
                edgecolors=edgecolor,
                linewidths=2.0,
                alpha=alpha,
                zorder=3,
            )
            if item["status"] != "complete":
                ax.annotate(
                    f"{item['effective_pairs']}/19",
                    (item["onset_jsd"], item["pitch_jsd"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7.5,
                    color=edgecolor,
                )

    ax.set_xlabel("Onset JSD (lower is better)")
    ax.set_ylabel("Pitch JSD (lower is better)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    setting_handles = []
    for key in ("i1-g1-p100", "i1-g3-p080", "i1-g3-p050", "i1-g3-p020"):
        style = SETTING_STYLE[key]
        setting_handles.append(
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
    setting_legend = ax.legend(handles=setting_handles, loc="upper left", frameon=True, title="Setting")
    ax.add_artist(setting_legend)

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

    note = "Color = model family; marker shape = rate-sweep setting; red-edge labels show completed pairs for incomplete runs."
    fig.text(0.02, 0.02, note, fontsize=8.5, color="#444444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot complete rate-sweep runs in JSD space.")
    parser.add_argument("--csv", required=True, help="by_run csv path")
    parser.add_argument("--out", required=True, help="output image path")
    parser.add_argument("--title", default="Unrolling Rate Sweep in JSD Space")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    plot(rows, Path(args.out), args.title)


if __name__ == "__main__":
    main()
