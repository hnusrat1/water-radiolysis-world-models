"""Plot archived checkpoint records without running a model."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

METHODS = [("compact32_active", "C32"), ("compact64_active", "C64"),
           ("moment_learned", "Learned moments")]
# Distinct hues plus stroke patterns for overlap and grayscale printing.
STYLES = {
    1701: ("#476C91", "-"),
    1702: ("#A43F21", "-"),
    1703: ("#23806C", (0, (3.8, 1.6))),
    1704: ("#735587", (0, (3.5, 1.3, 0.8, 1.3))),
    1705: ("#656565", (0, (1.2, 1.2))),
}


def plot(source, destination):
    traces = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fit = (row["method"], int(row["seed"]))
            traces[fit].append((int(row["epoch"]), float(row["training_loss"]) if
                                row["training_loss"] else None, float(row["validation_score"]),
                                int(row["selected"])))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.linewidth": 0.55, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(6.65, 3.45), sharex=True, sharey="row")
    fig.subplots_adjust(left=.098, right=.987, top=.925, bottom=.235, hspace=.18, wspace=.17)
    for col, (method, name) in enumerate(METHODS):
        train_ax, val_ax = axes[:, col]
        train_ax.set_title(name, weight="semibold", fontsize=9.1, pad=4.5)
        for seed in [1701, 1703, 1704, 1705, 1702]:
            color, style = STYLES[seed]
            records = traces[method, seed]
            train = [(epoch, loss) for epoch, loss, _, _ in records if loss is not None]
            train_ax.plot(*zip(*train), color=color, linestyle=style, linewidth=.95)
            val_ax.plot([r[0] for r in records], [r[2] for r in records],
                        color=color, linestyle=style, linewidth=.95)
            chosen = [r for r in records if r[3] == 1]
            if len(chosen) != 1:
                raise ValueError(f"Expected one selected score for {method} seed {seed}")
            epoch, _, score, _ = chosen[0]
            val_ax.plot(epoch, score, marker="o", ms=3.9, mfc="none", mec=color,
                        markeredgewidth=.85, linestyle="none", zorder=5)

        for ax, limits, ticks in [(train_ax, (.126, .345), [.13, .16, .20, .25, .32]),
                                   (val_ax, (.057, 1), [.06, .10, .20, .40, .80])]:
            ax.set_yscale("log")
            ax.set_ylim(limits)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FixedFormatter([f"{v:.2f}" for v in ticks]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_xlim(-2, 123)
            ax.set_xticks([0, 40, 80, 120])
            ax.tick_params(axis="both", length=2.3, width=.55, labelsize=7.5, pad=2.2)
            ax.grid(axis="y", alpha=.45, color="#CBCBCB", lw=.5)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Training loss", fontsize=8.5)
    axes[1, 0].set_ylabel("Validation score", fontsize=8.5)
    fig.text(.542, .115, "Epoch", ha="center", fontsize=8.7)
    legend = [Line2D([], [], color=color, linestyle=style, lw=1.4, label=str(seed))
              for seed, (color, style) in STYLES.items()]
    legend.append(Line2D([], [], marker="o", color="#353535", lw=0, mfc="none",
                         ms=4.4, label="Selected epoch"))
    fig.legend(handles=legend, loc="lower center", frameon=False, ncol=6,
               bbox_to_anchor=(.52, .008), fontsize=7.6,
               handlelength=1.75, handletextpad=.4, columnspacing=1.2)
    fig.savefig(destination, metadata={"Title": "200 eV training and validation trajectories",
                                       "Creator": "Matplotlib", "CreationDate": None,
                                       "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    plot(args.csv, args.pdf)
