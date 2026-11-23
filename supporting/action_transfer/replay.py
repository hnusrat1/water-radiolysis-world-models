"""Recalculate action-transfer estimates from the included per-family fixed-fit losses."""
import json
from pathlib import Path
import numpy as np


def summarize(data):
    families = data["families"]
    assert len(set(families)) == len(families) == 64
    assert set(data["strata"]) == {"unseen_delay", "three_pulse"}
    frozen = data["bootstrap"]
    draws = np.random.default_rng(frozen["seed"]).integers(0, len(families), (frozen["resamples"], len(families)))
    out = {}
    for stratum, groups in data["strata"].items():
        means = {}
        per_fit = {}
        for name, rows in groups.items():
            values = np.asarray(rows, dtype=np.float64)
            assert values.shape == (len(data["seeds"][name]), len(families))
            assert np.isfinite(values).all() and (values >= 0).all()
            means[name] = values.mean(axis=0)
            per_fit[name] = dict(zip(map(str, data["seeds"][name]), values.mean(axis=1).tolist()))
        contrasts = {}
        for comparator in ("C64", "C32"):
            difference = means["Learned moments"] - means[comparator]
            contrasts["Learned moments minus " + comparator] = dict(
                mean=float(difference.mean()),
                ci95=np.percentile(difference[draws].mean(axis=1), [2.5, 97.5]).tolist())
        out[stratum] = dict(families=64, cases_per_family=data["actions_per_family"][stratum],
                             raw_nmse={k:float(v.mean()) for k, v in means.items()},
                             raw_nmse_per_fit=per_fit, contrasts=contrasts)
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    result = summarize(json.loads((here / "family_losses.json").read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))
