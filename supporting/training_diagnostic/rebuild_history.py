"""Recover the published training curves from the original 15 JSON fit records."""
import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("compact32_active", "compact64_active", "moment_learned")
SEEDS = range(1701, 1706)
FIELDS = ("method", "seed", "epoch", "training_loss", "validation_score", "selected")


def rebuild():
    evidence = json.loads((HERE / "fit_diagnostic_evidence.json").read_text())
    fits = {(row["method"], row["seed"]): row for row in evidence["fits"]}
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(fits) != expected or len(evidence["fits"]) != len(expected):
        raise ValueError("Incorrect recorded fit roster")
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for method in METHODS:
        for seed in SEEDS:
            source = HERE / "training_records" / f"{method}_seed{seed}.json"
            raw = source.read_bytes()
            archived = fits[(method, seed)]
            if sha256(raw).hexdigest() != archived["training_record_sha256"]:
                raise ValueError(f"Source identity differs for {source.name}")
            record = json.loads(raw)
            if ((record["method"], record["seed"]) != (method, seed)
                    or record["protocol_sha256"] != evidence["protocol_sha256"]
                    or record["checkpoint_sha256"] != archived["checkpoint_sha256"]):
                raise ValueError(f"Fit provenance differs for {source.name}")
            best = min(record["records"], key=lambda row: row["validation_score"])
            if (best["epoch"],best["validation_score"]) != (archived["selected_epoch"],archived["selected_validation_score"]):
                raise ValueError(f"Selected checkpoint differs for {source.name}")
            for row in record["records"]:
                writer.writerow(dict(method=method, seed=seed, epoch=row["epoch"],
                                     training_loss=row["training_loss"], validation_score=row["validation_score"],
                                     selected=int(row["epoch"] == best["epoch"])))
    data = buf.getvalue().encode()
    if sha256(data).hexdigest() != evidence["trajectory_csv_sha256"] or data != (HERE / "fit_trajectories.csv").read_bytes():
        raise ValueError("Rebuilt training curves differ from the published CSV")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the regenerated CSV to this path")
    args = parser.parse_args()
    csv_bytes = rebuild()
    if args.output:
        args.output.write_bytes(csv_bytes)
    print(f"Verified {len(csv_bytes.splitlines()) - 1} training and validation records across 15 original fits.")
