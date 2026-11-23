"""Extract the two transfer-energy endpoints from the archived confirmation bank."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHOD_KEYS = {
    "moments": "refit_moment_learned",
    "C32": "refit_compact32_active",
    "C64": "refit_compact64_active",
}


def extract(study, output):
    hashes = json.loads((Path(__file__).parent / "data/upstream_sha256.json").read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((study / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Upstream file differs from the archived version: {name}")

    bank = study / "confirmation_bank"
    cases = json.loads((bank / "data/cases.json").read_bytes())
    by_id = {case["case_id"]: case for case in cases}
    forecast = json.loads((bank / "results/forecast_base.json").read_bytes())
    protocol = json.loads((bank / "protocol.json").read_bytes())["analysis"]
    if (protocol["bootstrap_seed"], protocol["bootstrap_replicates"]) != (202611142300, 4000):
        raise ValueError("Unexpected resampling procedure")

    output.mkdir(parents=True, exist_ok=True)
    for energy in (100, 400):
        split = f"energy{energy}"
        predictions = study / "evaluation/predictions" / f"{split}_base.npz"
        references = bank / "data" / f"labels_{split}.npz"
        with np.load(predictions, allow_pickle=False) as pred, np.load(references, allow_pickle=False) as ref:
            ids = ref["case_id"]
            np.testing.assert_array_equal(ids, pred["case_id"].astype(int))
            np.testing.assert_array_equal(ref["family_id"], pred["family_id"].astype(int))
            np.testing.assert_array_equal(ref["replicate_count"], np.full(ids.shape, 64))
            if ref["times_ns"][-1] != 20 or not np.all(pred["observation_time_ns"][:, -1] == 20):
                raise ValueError("The terminal time must be 20 ns")

            roster = [by_id[case_id] for case_id in ids]
            np.testing.assert_array_equal(ref["family_id"], [c["family_id"] for c in roster])
            if any(c["split"] != split for c in roster):
                raise ValueError("Wrong split in case roster")
            metadata = [c["action_metadata"] for c in roster]
            arrays = {
                "energy_eV": np.array(energy),
                "time_ns": ref["times_ns"][-1],
                "case_id": ids,
                "family_id": ref["family_id"],
                "pulse_time_ns": np.array([a["chemical_injection_time_ns"] for a in metadata]),
                "spacing_nm": np.array([a["incoming_spacing_nm"] for a in metadata]),
                "seeds": np.arange(1701, 1706),
                "species": ref["species"],
                "scale": pred["scale"],
            }
            for arm in ("forced", "passive"):
                counts = ref[f"{arm}_samples"][:, :, -1, :]
                np.testing.assert_array_equal(counts.mean(axis=1), ref[f"{arm}_mean"][:, -1, :])
                arrays[f"sbs_{arm}"] = counts
                for method, original in METHOD_KEYS.items():
                    recorded = forecast["strata"][split]["methods"][original]["seeds"]
                    np.testing.assert_array_equal(recorded, arrays["seeds"])
                    arrays[f"{method}_{arm}"] = pred[f"{original}_{arm}"][:, :, -1, :]
            np.savez_compressed(output / f"{split}.npz", **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Archived data/icwm/extension/strengthening_v1 directory")
    parser.add_argument("output", type=Path, help="Directory for the two endpoint files")
    args = parser.parse_args()
    extract(args.study, args.output)
