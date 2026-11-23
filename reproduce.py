"""Reconstruct and verify the manuscript's 20 ns transfer comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from primary.reference.replay_family_contrasts import summarize
from primary.score_endpoints import score_endpoints

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "primary"
REFERENCE = PRIMARY / "reference"
METHODS = ("moments", "C32", "C64")


def verify_inputs():
    for line in (PRIMARY / "SHA256SUMS").read_text().splitlines():
        expected, path = line.split("  ", 1)
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"File differs from the archived release: {path}")


def check_close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=label)


def replay():
    verify_inputs()
    families, per_fit = score_endpoints()
    archived = json.loads((REFERENCE / "family_losses.json").read_bytes())
    headline = json.loads((ROOT / "supporting/headline_replay/inputs.json").read_bytes())
    means = {}
    for energy in ("100", "400"):
        split = f"energy{energy}"
        observed = families["strata"][split]
        saved = archived["strata"][split]
        if observed["family_ids"] != saved["family_ids"]:
            raise ValueError(f"Family roster mismatch at {energy} eV")
        for field in ("nmse_by_method", "paired_differences"):
            for name, values in saved[field].items():
                check_close(observed[field][name], values, f"{energy} eV {field} {name}")
        expected_by_fit = headline["data"]["by_energy_eV"][energy]["per_fit_raw_response_nmse"]
        means[energy] = {}
        for name in METHODS:
            original_name = "learned_moments" if name == "moments" else name
            check_close(per_fit[split][name], expected_by_fit[original_name], f"{energy} eV {name} per fit")
            means[energy][name] = float(np.mean(observed["nmse_by_method"][name]))

    expected = json.loads((REFERENCE / "expected_contrasts.json").read_bytes())
    contrasts = {}
    for comparator, key in (("C64", "learned_moments_minus_C64_primary"),
                            ("C32", "learned_moments_minus_C32_secondary")):
        result = summarize(families, comparator)
        check_close(result["mean"], expected[key]["mean"], key)
        check_close(result["ci95"], expected[key]["ci95"], f"{key} 95% interval")
        contrasts[key] = result

    return {
        "nmse_by_energy_eV": means,
        "contrasts": contrasts,
        "relative_reduction_vs_C64_at_400_eV_percent": 100 * (1 - means["400"]["moments"] / means["400"]["C64"]),
        "per_fit_nmse": per_fit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print full-precision results as JSON")
    args = parser.parse_args()
    result = replay()
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return

    print("20 ns pulse-response NMSE averaged over five fixed fits")
    print("Energy       Learned moments       C32       C64")
    for energy, values in result["nmse_by_energy_eV"].items():
        print(f"{energy} eV             {values['moments']:.4f}    {values['C32']:.4f}    {values['C64']:.4f}")
    print()
    for key, label in (("learned_moments_minus_C64_primary", "Primary, learned moments minus C64"),
                       ("learned_moments_minus_C32_secondary", "Secondary, learned moments minus C32")):
        contrast = result["contrasts"][key]
        low, high = contrast["ci95"]
        print(f"{label}: {contrast['mean']:.4f} (paired 95% CI {low:.4f} to {high:.4f})")
    reduction = result["relative_reduction_vs_C64_at_400_eV_percent"]
    print(f"Relative NMSE reduction at 400 eV versus C64: {reduction:.1f}%")
    print("Verified against the archived family losses, fit scores, and paired intervals.")


if __name__ == "__main__":
    main()
