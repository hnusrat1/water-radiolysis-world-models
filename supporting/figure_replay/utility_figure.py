"""Draw the appendix yield-cost figure from the same CPU records as the yield table."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "utility_cpu.json"
SOURCE_SHA = "c1774deff42ed7adec69e820806da33bb5a767faa0215c9d82bbde7d0e8ad9c2"
METHODS = {
    "native_census": ("Native census", "#212428", "D"),
    "native_only": ("Native sample", "#62666a", "s"),
    "counts": ("Counts only", "#92682c", "^"),
    "moment_physics": ("Physical moments", "#998725", "v"),
    "refit_compact32_active": ("C32", "#3374a7", "P"),
    "refit_compact64_active": ("C64", "#bd5064", "X"),
    "refit_moment_learned": ("Learned moments", "#177757", "o"),
}

def extract(source):
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA:
        raise ValueError("Unexpected cost records")
    observations = json.loads(content)["passive_utility_acquisition_charged"]
    rows = []
    for key, obj in sorted(observations.items()):
        model = key.rsplit("_m", 1)[0]
        assert model in METHODS
        error = obj.get("accuracy_first_fit_finite_bank_MSE", obj.get("accuracy_finite_bank_MSE"))
        cost = obj["cost"]["fresh_process_CPU_seconds"]["source_acquisition_charged"]
        rows.append(dict(model=model, paths=obj["continuations"], mse=error["mean"],
                         ci=error["ci95"], cpu=cost["median"], quartiles=cost["quartiles"]))
    return rows

def render(rows, output):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,2,sharey=True,figsize=(7.15,3.4))
    fig.subplots_adjust(left=.12,right=.985,top=.88,bottom=.30,wspace=.16)
    for ax, budget in zip(axes, [64,512]):
        for method, (label,color,marker) in METHODS.items():
            row, = [row for row in rows if row["model"]==method and row["paths"]==budget]
            x,y = row["cpu"],row["mse"]
            lo,hi = row["ci"]
            q1,q3 = row["quartiles"]
            ax.errorbar(x,y,[[y-lo],[hi-y]],[[x-q1],[q3-x]],fmt=marker,color=color,
                        lw=1,ms=5,capsize=1.5,label=label)
        ax.set_yscale("log")
        ax.set_title(f"{budget} simulation paths")
        ax.grid(axis="y",alpha=.20)
        ax.set_xlabel("CPU including source acquisition (s)")
    axes[0].set_ylabel("Mean-yield MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=7.8)
    fig.savefig(output, metadata={"Title":"Mean passive OH yield error and CPU cost","CreationDate":None,"ModDate":None})
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-source",type=Path)
    args = parser.parse_args()
    rows = json.loads(DATA.read_text())
    if args.verify_source and rows != extract(args.verify_source):
        raise ValueError("Packaged data differ from archived records")
    render(rows, args.output)
