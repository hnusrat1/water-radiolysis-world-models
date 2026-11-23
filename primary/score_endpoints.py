"""Score 20 ns predictions against individual step-by-step simulation endpoints."""

from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / "data"
METHODS = ("moments", "C32", "C64")
SPECIES = ("e_aq", "OH", "H", "H2", "H3O+", "OH-", "H2O2")


def score_energy(data, energy):
    family = data["family_id"]
    ids, counts = np.unique(family, return_counts=True)
    case_ids = data["case_id"]
    if len(ids) != 64 or not np.all(counts == 4) or case_ids.shape != (256,) or np.unique(case_ids).size != 256:
        raise ValueError(f"Expected four distinct cases for each of 64 families at {energy} eV")
    if data["energy_eV"].item() != energy or data["time_ns"].item() != 20:
        raise ValueError("Unexpected energy or observation time")
    if data["species"].tolist() != list(SPECIES) or data["seeds"].tolist() != list(range(1701, 1706)):
        raise ValueError("Unexpected species or fit order")
    times, spacing = data["pulse_time_ns"], data["spacing_nm"]
    for source_id in ids:
        mask = family == source_id
        if set(zip(times[mask], spacing[mask])) != {(1, 0), (1, 50), (10, 0), (10, 50)}:
            raise ValueError(f"Unexpected pulse schedule for family {source_id}")

    reference = {}
    for arm in ("forced", "passive"):
        paths = data[f"sbs_{arm}"]
        if paths.shape != (256, 64, 7) or not np.issubdtype(paths.dtype, np.integer) or np.any(paths < 0):
            raise ValueError(f"Invalid SBS endpoint counts on the {arm} branch")
        reference[arm] = paths.mean(axis=1)

    observed_response = reference["forced"] - reference["passive"]
    scale = data["scale"]
    if scale.shape != (7,) or not np.isfinite(scale).all() or np.any(scale < 1):
        raise ValueError("Invalid training-set species scales")

    def average_actions(values):
        return np.array([values[family == source_id].mean() for source_id in ids])

    case_losses, per_fit = {}, {}
    for method in METHODS:
        forced, passive = data[f"{method}_forced"], data[f"{method}_passive"]
        for branch in (forced, passive):
            if branch.shape != (5, 256, 7) or not np.isfinite(branch).all():
                raise ValueError(f"Invalid {method} forecasts")
        errors = (forced - passive) - observed_response[None, :, :]
        # Preserve the study's reduction order for the paired family analysis.
        mean_squared_error = (errors ** 2).mean(axis=0)
        case_losses[method] = (mean_squared_error / scale ** 2).mean(axis=1)
        losses_by_fit = ((errors / scale) ** 2).mean(axis=2)
        per_fit[method] = [float(average_actions(row).mean()) for row in losses_by_fit]

    families = {
        "family_ids": ids.astype(str).tolist(),
        "nmse_by_method": {name: average_actions(loss).tolist() for name, loss in case_losses.items()},
        "paired_differences": {
            f"moments_minus_{name}": average_actions(case_losses["moments"] - case_losses[name]).tolist()
            for name in ("C32", "C64")
        },
    }
    return families, per_fit


def score_endpoints(data_dir=DATA):
    """Return family-level bootstrap inputs and per-fit NMSE by energy and method.

    Per-fit lists use the fixed seed order 1701, 1702, 1703, 1704, 1705.
    """
    family_data = {
        "schema": "fresh-base-terminal-family-losses-v1",
        "scope": "Seven-species 20 ns raw response NMSE averaged over five fixed fits and four actions per family.",
        "bootstrap_seed": 202611142300,
        "draws": 4000,
        "strata": {},
    }
    per_fit = {}
    for energy in (100, 400):
        split = f"energy{energy}"
        with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as data:
            family_data["strata"][split], per_fit[split] = score_energy(data, energy)
    return family_data, per_fit
