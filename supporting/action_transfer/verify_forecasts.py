"""Regenerate the archived 200 eV temporal transfer predictions on CPU."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

from replay import summarize

HERE = Path(__file__).resolve().parent
DATA = "dataset/main/"
METHODS = {"C32": "c32", "C64": "c64", "Learned moments": "moments", "Physical moments": None}
SCHEDULES = {"simultaneous_three_clouds": "unseen_delay", "three_pulse": "three_pulse"}


def read(path):
    return json.loads(path.read_text())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def compare(actual, expected, tolerance, name):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=False, err_msg=name)
    return float(np.max(np.abs(np.asarray(actual) - expected)))


def load_public_data(archive, plan, need_labels):
    case_file = DATA + "cases_temporaltest.json"
    with ZipFile(archive) as z:
        raw = z.read("icwm/" + case_file)
        require(sha256(raw).hexdigest() == plan["archive_file_sha256"][case_file], "Temporal case hash differs")
        cases = json.loads(raw)
        if not need_labels:
            return cases, None
        labels = z.read("icwm/" + DATA + "labels_temporaltest.npz")
    require(sha256(labels).hexdigest() == plan["native_label_sha256"], "Temporal reference hash differs")
    ids = np.array([str(case["case_id"]) for case in cases])
    with np.load(BytesIO(labels), allow_pickle=False) as native:
        label_ids = native["case_id"].astype(str)
        lookup = {case_id: i for i, case_id in enumerate(label_ids)}
        require(len(lookup) == len(cases) and set(lookup) == set(ids), "Unmatched case IDs")
        ix = [lookup[case_id] for case_id in ids]
        require(np.all(native["replicate_count"] == 64), "Expected 64 native paths per branch")
        for arm in ("forced", "passive"):
            samples = native[f"{arm}_samples"]
            require(samples.shape[1] == 64, "Incorrect replicate axis")
            means = samples[ix, :, -1].mean(axis=1)
            np.testing.assert_array_equal(means, native[f"{arm}_mean"][ix, -1])
            if arm == "forced":
                target = means.astype(np.float64)
            else:
                target -= means
    return cases, target


def verify_estimates(calculated, reference):
    if isinstance(reference, dict):
        require(set(calculated) == set(reference), "Different keys in published statistics")
        for key in reference:
            verify_estimates(calculated[key], reference[key])
    else:
        compare(calculated, reference, 1e-9, "Mismatch in published statistics")


@torch.no_grad()
def verify(archive, runtime, full):
    require((runtime/"radiolysis/inference.py").is_file(), "Supply the repository's inference directory with --inference")
    sys.path.insert(0, str(runtime))
    from radiolysis import load_predictor
    from radiolysis.data_contract import cases_to_arrays
    from radiolysis.recurrent import make_model
    from radiolysis.solver import SolverConfig, rollout
    from verify import input_batch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan_path = HERE/"plan.json"
    plan = read(plan_path)
    specimen = read(HERE/"three_pulse_reference.json")
    require(sha256(plan_path.read_bytes()).hexdigest() == specimen["original_plan_sha256"], "Temporal plan identity differs")
    config = read(runtime/"config.json")
    require(config["solver"] == plan["controls"]["base"] and config["species_scales"] == plan["scales"], "Controls or scales differ")
    original_ids = set(plan["checkpoint_sha256"].values())
    require(all(s["source_checkpoint_sha256"] in original_ids for s in config["models"]), "Model provenance differs")
    cases, target = load_public_data(archive, plan, full)
    arrays = cases_to_arrays(cases)
    schedules = np.array([SCHEDULES[c["action_metadata"]["pattern"]] for c in cases])
    families = arrays["family_id"].astype(str)
    block_size = plan["batch"]
    require((len(cases), len(set(families)), block_size) == (576, 64, 128), "Temporal roster differs")
    ids = list(arrays["case_id"].astype(str))
    ref_row = ids.index(specimen["case_id"])
    ref_case = cases[ref_row]
    require(schedules[ref_row] == "three_pulse" and families[ref_row] == specimen["family_id"], "Wrong smoke-test case")
    require([a["time_ns"] for a in ref_case["actions"]] == specimen["schedule_ns"], "Wrong pulse times")

    reference = read(HERE/"family_losses.json")
    family_order = reference["families"]
    require(family_order == plan["family_ids"] and set(families) == set(family_order), "Family mapping differs")
    calculated = {k:v for k,v in reference.items() if k != "strata"}
    calculated["strata"] = {s: {m: [] for m in METHODS} for s in SCHEDULES.values()}
    peak_loss = peak_trace = 0.0
    all_models = [(m, seed) for m in METHODS for seed in reference["seeds"][m]]
    roster = all_models if full else [("Learned moments", 1701)]
    for method, seed in roster:
        prefix = METHODS[method]
        model_id = f"{prefix}-{seed}" if prefix else "moment_physics"
        if prefix:
            model, controls = load_predictor(model_id)
        else:
            model = make_model("moment_physics", capture_scale=config["capture_scale"]).double().eval()
            controls = SolverConfig(**config["solver"])
        # Adaptive integration uses shared step decisions within each original batch.
        starts = range(0, len(cases), block_size) if full else [(ref_row // block_size) * block_size]
        endpoints = []
        for start in starts:
            rows = np.arange(start, min(start + block_size, len(cases)))
            batch = input_batch(arrays, rows)
            counts = {}
            for arm in ("passive", "forced"):
                solved = rollout(model, batch, forced=arm == "forced", config=controls)
                counts[arm] = solved["counts"][:, -1].numpy()
                if model_id == specimen["model"] and start <= ref_row < start + len(rows):
                    for field, source in [(arm, "counts"), (arm + "_water_ledger", "water_ledger")]:
                        observed = solved[source][ref_row - start].numpy()
                        expected = np.asarray(specimen["trajectories"][field])
                        peak_trace = max(peak_trace, compare(observed, expected, 1e-8, model_id+" "+field))
            endpoints.append(counts["forced"] - counts["passive"])
        if full:
            errors = np.mean(((np.concatenate(endpoints) - target) / plan["scales"])**2, axis=1)
            fit_index = reference["seeds"][method].index(seed)
            for schedule, actions in reference["actions_per_family"].items():
                losses = []
                for family in family_order:
                    mask = (families == family) & (schedules == schedule)
                    require(mask.sum() == actions, "Incomplete schedule within a family")
                    losses.append(float(errors[mask].mean()))
                stored = reference["strata"][schedule][method][fit_index]
                peak_loss = max(peak_loss, compare(losses, stored, 1e-9, model_id+" "+schedule))
                calculated["strata"][schedule][method].append(losses)
        print("Verified", model_id, flush=True)
    if full:
        verify_estimates(summarize(calculated), read(HERE/"expected_results.json"))
    return dict(cloud_families=len(family_order), verified_fixed_predictors=len(roster),
                seed1701_three_pulse_case_id=specimen["case_id"], max_smoke_trajectory_difference=peak_trace,
                paired_forecasts=len(cases)*len(roster) if full else block_size,
                family_losses_checked=len(family_order)*2*len(roster) if full else 0,
                max_family_loss_difference=peak_loss if full else None,
                summary_estimates_and_intervals_checked=full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="Path to ICWM_v13_ablation_training_and_evaluation_inputs.zip")
    parser.add_argument("--inference", type=Path, default=HERE.parents[1]/"inference", help="Public inference code, configuration and safe weights")
    parser.add_argument("--full", action="store_true", help="Check both schedules for 64 families and all sixteen predictors")
    parser.add_argument("--json", type=Path, help="Write the machine-readable verification receipt")
    args = parser.parse_args()
    result = verify(args.inputs.resolve(), args.inference.resolve(), args.full)
    text = json.dumps(result, indent=2) + "\n"
    print(text)
    if args.json:
        args.json.write_text(text)
