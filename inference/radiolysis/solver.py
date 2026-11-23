"""Positive SSPRK3 integration of the frozen closures' continuous vector field.

This module imports, but never changes, the original model implementation.
The original encoder, hazards, source jumps, reaction network and geometry
closures are retained.  It replaces the first-order numerical update only.
Gaussian moments are integrated as extensive moments, eliminating the
coordinate singularity in normalized means when a species is born.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .recurrent import (ChemicalState, CompactEquivariantClosure,
                       GaussianMomentClosure, outer, pack_symmetric)


@dataclass(frozen=True)
class SolverConfig:
    max_step_ns: float = 0.125
    max_fraction: float = 0.2
    rtol: float = 1e-5
    atol: float = 1e-7
    adaptive: bool = True
    minimum_step_ns: float = 1e-10
    max_attempts: int = 100000

    def __post_init__(self):
        if self.max_step_ns <= 0 or not 0 < self.max_fraction < 1:
            raise ValueError("positive maximum step and fraction in (0,1) required")
        if self.rtol <= 0 or self.atol <= 0 or self.minimum_step_ns <= 0:
            raise ValueError("positive error and minimum-step controls required")

    def dictionary(self):
        return asdict(self)


def _pack(model, state: ChemicalState) -> Tensor:
    memory = state.memory
    if isinstance(model, GaussianMomentClosure):
        mu, second = model._split(memory)
        memory = model._join(mu * state.counts[..., None],
                             second * state.counts[..., None, None])
    return torch.cat((state.counts, memory, state.water_ledger[:, None]), -1)


def _state(model, y: Tensor, origin: Tensor, clock: Tensor) -> ChemicalState:
    counts, memory = y[:, :7], y[:, 7:-1]
    if isinstance(model, GaussianMomentClosure):
        first, second = model._split(memory)
        # Zero mass has zero extensive moments. Its conditional geometry is
        # undefined and irrelevant to every hazard requiring that species.
        denominator = torch.where(counts > 0, counts, torch.ones_like(counts))
        memory = model._join(first / denominator[..., None],
                            second / denominator[..., None, None])
    return ChemicalState(counts, memory, origin, y[:, -1], clock.to(y.dtype))


def vector_field(model, y: Tensor, origin: Tensor, clock: Tensor,
                 fraction: float = 0.2):
    """Return the original infinitesimal closure and a positive FE step bound.

    Counts/water follow the unchanged stoichiometric rates. Compact latent
    derivatives are the dt->0 limits of _diffuse_memory and _latent_flow.
    Gaussian extensive derivatives are the dt->0 limit of uniform reactant
    removal and Gaussian product insertion, plus exact diffusion.
    """
    state = _state(model, y, origin, clock)
    rates = model._hazards(state)
    if not bool(torch.isfinite(rates).all()) or bool((rates < 0).any()):
        raise FloatingPointError("frozen hazard is not finite and nonnegative")
    consumed = rates @ model.reactants.T
    inf = torch.full_like(consumed, math.inf)
    safe = torch.where(consumed > 0,
                       state.counts / torch.where(consumed > 0, consumed,
                                                  torch.ones_like(consumed)), inf)
    bound = fraction * safe.min(-1).values
    dn = rates @ model.stoich.T
    dw = rates @ model.water_stoich
    dm = torch.zeros_like(state.memory)

    if isinstance(model, CompactEquivariantClosure):
        v, tensor, scalar = model._split(state.memory)
        output = model.flow_head(model._features(state))
        i = model.nv * model.nv
        j = i + model.nv
        k = j + model.nt
        mix = output[:, :i].reshape(-1, model.nv, model.nv).softmax(-1)
        vrate = F.softplus(output[:, i:j] - 4)
        trate = F.softplus(output[:, j:k] - 4)
        dv = vrate[..., None] * (torch.einsum("bvw,bwj->bvj", mix, v) - v)
        target = outer(v).mean(1, keepdim=True).expand(-1, model.nt, -1, -1)
        dtensor = trate[..., None, None] * (target - tensor)
        _, weights = model._weights()
        weighted = state.counts[:, None, :] * weights[None]
        denominator = weighted.sum(-1)
        diffusion = ((weighted * model.diffusion).sum(-1)
                     / denominator.clamp_min(torch.finfo(y.dtype).eps))
        eye = torch.eye(3, dtype=y.dtype, device=y.device)
        dtensor = dtensor + (2 * diffusion[..., None, None]
                             / model.length_nm ** 2 * eye)
        ds = 0.05 * (torch.tanh(output[:, k:]) - scalar)
        dm = model._join(dv, dtensor, ds)
        # Each FE geometry step remains a positive mixture plus diffusion.
        rate_max = torch.cat((vrate, trate,
                              y.new_full((len(y), 1), 0.05)), -1).max(-1).values
        bound = torch.minimum(bound, fraction / rate_max)
    elif isinstance(model, GaussianMomentClosure):
        _, product_mu, product_second, mu, _ = model._encounters(state)
        _, second = model._split(state.memory)
        df = (-mu * consumed[..., None]
              + torch.einsum("br,sr,bri->bsi", rates, model.products, product_mu))
        dq = (-second * consumed[..., None, None]
              + torch.einsum("br,sr,brij->bsij", rates, model.products,
                             product_second))
        eye = torch.eye(3, dtype=y.dtype, device=y.device)
        dq = dq + (2 * state.counts[..., None, None]
                   * model.diffusion[None, :, None, None]
                   / model.length_nm ** 2 * eye)
        dm = model._join(df, dq)
    derivative = torch.cat((dn, dm, dw[:, None]), -1)
    if not bool(torch.isfinite(derivative).all()):
        raise FloatingPointError("nonfinite frozen continuous vector field")
    return derivative, bound.detach().to(torch.float64)


def _ssprk3(model, y, origin, clock, h, fraction):
    """SSPRK(3,3), with nonautonomous stage times t,t+h,t+h/2.

    No extrapolated state and no individual-species clipping are used.
    A rejected FE stage takes a zero step, keeping later trial stages valid.
    """
    failed = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    minimum_bound = torch.full_like(h, math.inf)

    def forward(z, t):
        nonlocal failed, minimum_bound
        derivative, bound = vector_field(model, z, origin, t, fraction)
        bad = (h > bound * (1 + 1e-12)) & (h > 0)
        failed = failed | bad
        minimum_bound = torch.minimum(minimum_bound, bound)
        effective = torch.where(bad, torch.zeros_like(h), h).to(y.dtype)
        return z + effective[:, None] * derivative

    first = forward(y, clock)
    # Increment form is algebraically the same positive convex combination.
    # It preserves an unchanged float32 component exactly and avoids adding
    # two large rounded fractions of every count/ledger at every substep.
    second = y + 0.25 * (forward(first, clock + h) - y)
    third = y + (2 / 3) * (forward(second, clock + h / 2) - y)
    third = torch.where((h > 0)[:, None], third, y)
    return third, ~failed, minimum_bound


def evolve(model, state: ChemicalState, duration_ns, config: SolverConfig):
    y = _pack(model, state)
    duration = torch.as_tensor(duration_ns, device=y.device,
                               dtype=torch.float64).expand(len(y))
    if bool((duration < -1e-7).any()):
        raise ValueError("negative evolution duration")
    remaining = duration.clamp_min(0)
    target = state.time_ns.to(torch.float64) + remaining
    proposed = remaining.new_full(remaining.shape, config.max_step_ns)
    statistics = dict(attempts=0, accepted_case_steps=0, rejected_case_steps=0,
                      positivity_rejected_case_steps=0, rhs_evaluations=0,
                      smallest_step_ns=math.inf, maximum_accepted_error_ratio=0.)
    while bool((remaining > 2e-13).any()):
        if statistics["attempts"] >= config.max_attempts:
            raise RuntimeError("SSPRK3 exceeded maximum attempts")
        h = torch.minimum(remaining, proposed)
        clock = target - remaining
        coarse, positive, bound = _ssprk3(model, y, state.origin_nm, clock, h,
                                         config.max_fraction)
        statistics["rhs_evaluations"] += 3
        if config.adaptive:
            half, ok1, b1 = _ssprk3(model, y, state.origin_nm, clock, h / 2,
                                   config.max_fraction)
            fine, ok2, b2 = _ssprk3(model, half, state.origin_nm, clock + h / 2,
                                   h / 2, config.max_fraction)
            statistics["rhs_evaluations"] += 6
            positive = positive & ok1 & ok2
            bound = torch.minimum(bound, torch.minimum(2 * b1, 2 * b2))
            # A third-order method has fine-solution local error estimated by
            # (fine-coarse)/(2**3-1). Do not extrapolate: positivity is retained.
            tolerance = config.atol + config.rtol * torch.maximum(y.abs(), fine.abs())
            error = ((fine - coarse).abs() / (7 * tolerance)).max(-1).values
            candidate = fine
            factor = (0.9 * error.detach().clamp_min(1e-12).pow(-0.25)).clamp(.2, 2.)
        else:
            candidate = coarse
            error = torch.zeros_like(h)
            factor = torch.full_like(h, 2.)
        active = h > 0
        accept = positive & (error <= 1) & active
        reject = ~accept & active
        if not bool(torch.isfinite(candidate).all()):
            raise FloatingPointError("nonfinite SSPRK3 trial")
        if bool((candidate[accept, :7] < -1e-10).any()):
            raise FloatingPointError("positive SSPRK3 trial produced negative counts")
        y = torch.where(accept[:, None], candidate, y)
        remaining = torch.where(accept, (remaining - h).clamp_min(0), remaining)
        statistics["attempts"] += 1
        statistics["accepted_case_steps"] += int(accept.sum())
        statistics["rejected_case_steps"] += int(reject.sum())
        statistics["positivity_rejected_case_steps"] += int((~positive & active).sum())
        if bool(accept.any()):
            statistics["smallest_step_ns"] = min(statistics["smallest_step_ns"],
                                                  float(h[accept].detach().min()))
            statistics["maximum_accepted_error_ratio"] = max(
                statistics["maximum_accepted_error_ratio"], float(error[accept].detach().max()))
        proposed = torch.minimum(h * factor.to(h.dtype),
                                  h.new_full(h.shape, config.max_step_ns))
        # Retain the most restrictive already observed FE bound. Without
        # this bound, a large growth factor repeatedly proposes and rejects
        # the same positivity-limited step on alternating iterations.
        proposed = torch.minimum(proposed, bound * .9)
        proposed = torch.where(~positive, torch.minimum(h * .5, bound * .8), proposed)
        if bool((reject & (proposed < config.minimum_step_ns)).any()):
            raise RuntimeError("SSPRK3 cannot satisfy error/positivity at minimum step")
        proposed = proposed.clamp_min(config.minimum_step_ns)
    if statistics["smallest_step_ns"] == math.inf:
        statistics["smallest_step_ns"] = 0.
    return _state(model, y, state.origin_nm, target), statistics


def _add_statistics(total, step):
    for key in ("attempts", "accepted_case_steps", "rejected_case_steps",
                "positivity_rejected_case_steps", "rhs_evaluations"):
        total[key] = total.get(key, 0) + step[key]
    if step["smallest_step_ns"]:
        total["smallest_step_ns"] = min(total.get("smallest_step_ns", math.inf),
                                         step["smallest_step_ns"])
    total["maximum_accepted_error_ratio"] = max(total.get("maximum_accepted_error_ratio", 0.),
                                                step["maximum_accepted_error_ratio"])


def rollout(model, batch: dict[str, Any], *, forced=True,
            config: SolverConfig | None = None):
    """Drop-in count/ledger/state forecast; input-only, no label access.

    Model dtype determines actual state/parameter precision and must be
    reported. Controls and RHS evaluation counters are returned separately.
    Pulses are exact original jump calls at their declared absolute clocks.
    """
    config = config or SolverConfig()
    state = model.encode(batch["initial_xyz_nm"], batch["initial_species"],
                         batch["initial_mask"], batch["initial_time_ns"])
    times = model._t(batch["observation_time_ns"])
    pulse_times = model._t(batch["pulse_time_ns"])
    valid = torch.as_tensor(batch["pulse_valid"], device=times.device, dtype=torch.bool)
    done = torch.zeros_like(valid)
    if times.ndim != 2 or times.shape[0] != len(state.counts):
        raise ValueError("observation times must have [B,T] shape")
    if bool((times[:, 1:] <= times[:, :-1]).any()):
        raise ValueError("observation times must increase")
    expected = model.conserved_totals(state)
    predictions, ledgers = [], []
    statistics = dict(maximum_atom_charge_drift=0., minimum_count=math.inf,
                      solver="positive-ssprk3", adaptive_step_doubling=config.adaptive,
                      configuration=config.dictionary(), dtype=str(state.counts.dtype),
                      state_bytes=model.storage(precision_bytes=state.counts.element_size())["state_bytes"])
    for j in range(times.shape[1]):
        target = times[:, j]
        if forced:
            for k in range(pulse_times.shape[1]):
                due = valid[:, k] & ~done[:, k] & (pulse_times[:, k] <= target + 1e-7)
                if bool(due.any()):
                    duration = torch.where(due, pulse_times[:, k] - state.time_ns,
                                           torch.zeros_like(target))
                    state, step = evolve(model, state, duration, config)
                    _add_statistics(statistics, step)
                    before = model.conserved_totals(state)
                    state = model.jump(state, batch["pulse_xyz_nm"][:, k],
                                       batch["pulse_species"][:, k], batch["pulse_mask"][:, k], due)
                    expected = expected + model.conserved_totals(state) - before
                    done[:, k] = done[:, k] | due
        state, step = evolve(model, state, target - state.time_ns, config)
        _add_statistics(statistics, step)
        predictions.append(state.counts)
        ledgers.append(state.water_ledger)
        statistics["maximum_atom_charge_drift"] = max(statistics["maximum_atom_charge_drift"],
            float((model.conserved_totals(state) - expected).detach().abs().max()))
        statistics["minimum_count"] = min(statistics["minimum_count"], float(state.counts.detach().min()))
    if "smallest_step_ns" not in statistics:
        statistics["smallest_step_ns"] = 0.
    return dict(counts=torch.stack(predictions, 1), water_ledger=torch.stack(ledgers, 1),
                state=state, statistics=statistics)
