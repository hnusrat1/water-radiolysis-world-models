"""Load a selected fit and forecast the passive and pulse branches on CPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from .data_contract import cases_to_arrays, SPECIES_NAMES
from .models import make_model
from .solver import SolverConfig, rollout

ROOT = Path(__file__).resolve().parent.parent
INPUT_KEYS = (
    "initial_xyz_nm", "initial_species", "initial_mask", "initial_time_ns",
    "pulse_xyz_nm", "pulse_species", "pulse_mask", "pulse_time_ns", "pulse_valid",
    "observation_time_ns",
)


def load_predictor(checkpoint="moments-1701"):
    """Return the evaluated PyTorch module and the study's solver controls."""
    config = json.loads((ROOT / "config.json").read_text())
    specs = {s["id"]: s for s in config["models"]}
    if checkpoint not in specs:
        raise ValueError(f"Unknown checkpoint {checkpoint!r}. Choices: {', '.join(specs)}")
    spec = specs[checkpoint]
    weight_file = ROOT / spec["weights"]
    if hashlib.sha256(weight_file.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Weight checksum mismatch for {checkpoint}")
    model = make_model(spec["method"], hidden=config["hidden"], capture_scale=config["capture_scale"]).double()
    model.load_state_dict(load_file(str(weight_file), device="cpu"), strict=True)
    return model.eval(), SolverConfig(**config["solver"])


@torch.no_grad()
def forecast(case, model, controls):
    """Return the full expected-count trajectories for one molecular-cloud case."""
    arrays = cases_to_arrays([case])
    batch = {}
    for key in INPUT_KEYS:
        a = arrays[key]
        if np.issubdtype(a.dtype, np.bool_):
            dtype = torch.bool
        elif np.issubdtype(a.dtype, np.integer):
            dtype = torch.long
        else:
            dtype = torch.float64
        batch[key] = torch.as_tensor(a, dtype=dtype)
    passive = rollout(model, batch, forced=False, config=controls)
    forced = rollout(model, batch, forced=True, config=controls)
    return {
        "species": list(SPECIES_NAMES),
        "times_ns": list(case["times_ns"]),
        "passive": passive["counts"][0].tolist(),
        "forced": forced["counts"][0].tolist(),
        "response": (forced["counts"][0] - passive["counts"][0]).tolist(),
        "passive_water_ledger": passive["water_ledger"][0].tolist(),
        "forced_water_ledger": forced["water_ledger"][0].tolist(),
    }
