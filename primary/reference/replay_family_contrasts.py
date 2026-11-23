"""Reproduce the paired transfer contrasts from the supplied family-level losses."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(data, comparator):
    rng = np.random.default_rng(data["bootstrap_seed"])
    means = []
    bootstrap_means = []
    for energy in ("energy100", "energy400"):
        stratum = data["strata"][energy]
        family_ids = stratum["family_ids"]
        moments = np.asarray(stratum["nmse_by_method"]["moments"], dtype=np.float64)
        baseline = np.asarray(stratum["nmse_by_method"][comparator], dtype=np.float64)
        contrast = np.asarray(
            stratum["paired_differences"]["moments_minus_" + comparator], dtype=np.float64
        )
        if len(set(family_ids)) != 64 or len(family_ids) != 64:
            raise ValueError(f"Expected 64 distinct source families at {energy}")
        if any(values.shape != (64,) for values in (moments, baseline, contrast)):
            raise ValueError(f"Unexpected family-loss shape at {energy}")
        if not np.allclose(moments - baseline, contrast, atol=1e-12, rtol=0):
            raise ValueError(f"Paired differences disagree with the separate losses at {energy}")
        selection = rng.integers(0, len(contrast), size=(data["draws"], len(contrast)))
        bootstrap_means.append(contrast[selection].mean(axis=1))
        means.append(contrast.mean())
    draws = np.mean(bootstrap_means, axis=0)
    return {"mean": float(np.mean(means)), "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "losses", nargs="?", type=Path, default=Path(__file__).with_name("family_losses.json")
    )
    args = parser.parse_args()
    data = json.loads(args.losses.read_text(encoding="utf-8"))
    if data["draws"] != 4000 or data["bootstrap_seed"] != 202611142300:
        raise ValueError("The resampling settings differ from those used in the study")
    results = {
        "learned_moments_minus_C64_primary": summarize(data, "C64"),
        "learned_moments_minus_C32_secondary": summarize(data, "C32"),
    }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
