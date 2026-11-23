"""Compare freshly computed 20 ns branch counts with the stored confirmation forecasts."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from radiolysis import load_predictor
from radiolysis.data_contract import SPECIES_NAMES, cases_to_arrays
from radiolysis.inference import INPUT_KEYS
from radiolysis.solver import rollout


HERE = Path(__file__).resolve().parent
ENERGIES = (100, 400)
METHODS = {"moments": "moments", "C32": "c32", "C64": "c64"}
SEEDS = tuple(range(1701, 1706))
BATCH_SIZE = 128
ABSOLUTE_TOLERANCE = 1e-8


def load_cases():
    cases = []
    for energy in ENERGIES:
        with gzip.open(HERE / "cases" / f"energy{energy}.json.gz", "rt", encoding="utf-8") as f:
            cases.extend(json.load(f))
    # The original evaluator padded all cases before selecting rows by energy.
    # Both particle-count maxima occur in the supplied 400 eV group.
    return cases_to_arrays(cases)


def input_batch(arrays, rows):
    batch = {}
    for key in INPUT_KEYS:
        values = arrays[key][rows]
        tensor = torch.from_numpy(values)
        if tensor.is_floating_point():
            tensor = tensor.double()
        batch[key] = tensor
    return batch


def max_difference(actual, expected, description):
    np.testing.assert_allclose(
        actual, expected, atol=ABSOLUTE_TOLERANCE, rtol=0., equal_nan=False,
        err_msg=description,
    )
    return float(np.max(np.abs(actual - expected)))


@torch.no_grad()
def verify(full):
    started = perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays = load_cases()
    pairs = 0
    largest_branch = 0.
    largest_response = 0.
    for energy in ENERGIES if full else (400,):
        rows = np.flatnonzero(arrays["split"] == f"energy{energy}")
        archive = HERE.parent / "primary" / "data" / f"energy{energy}.npz"
        with np.load(archive, allow_pickle=False) as stored:
            np.testing.assert_array_equal(arrays["case_id"][rows], stored["case_id"].astype(str))
            np.testing.assert_array_equal(arrays["family_id"][rows], stored["family_id"].astype(str))
            np.testing.assert_array_equal(stored["species"].astype(str), SPECIES_NAMES)
            np.testing.assert_array_equal(stored["seeds"], SEEDS)
            if len(rows) != 256 or not np.all(arrays["observation_time_ns"][rows, -1] == stored["time_ns"]):
                raise ValueError(f"Unexpected case roster or terminal query at {energy} eV")
            for method in METHODS if full else ("moments",):
                target = {arm: stored[f"{method}_{arm}"] for arm in ("passive", "forced")}
                for fit_index, seed in enumerate(SEEDS if full else (1701,)):
                    name = f"{METHODS[method]}-{seed}"
                    model, controls = load_predictor(name)
                    for start in range(0, len(rows) if full else BATCH_SIZE, BATCH_SIZE):
                        block = slice(start, start + BATCH_SIZE)
                        batch = input_batch(arrays, rows[block])
                        actual = {}
                        for arm in ("passive", "forced"):
                            counts = rollout(model, batch, forced=(arm == "forced"), config=controls)["counts"]
                            actual[arm] = counts[:, -1, :].numpy()
                            error = max_difference(
                                actual[arm], target[arm][fit_index, block],
                                f"{energy} eV, {name}, {arm}, cases starting at row {start}",
                            )
                            largest_branch = max(largest_branch, error)
                        difference = max_difference(
                            actual["forced"] - actual["passive"],
                            target["forced"][fit_index, block] - target["passive"][fit_index, block],
                            f"{energy} eV, {name}, pulse response, cases starting at row {start}",
                        )
                        largest_response = max(largest_response, difference)
                        pairs += len(rows[block])
                    print(f"Verified {energy} eV, {name} ({pairs:,} paired forecasts total)", flush=True)
    print(f"Maximum absolute branch-count difference: {largest_branch:.12g} molecules")
    print(f"Maximum absolute response difference: {largest_response:.12g} molecules")
    print(f"Elapsed wall time: {perf_counter() - started:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Verify all 7,680 fit/case pairs.")
    verify(parser.parse_args().full)
