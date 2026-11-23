"""Replay the v5 headline summaries from the packaged observations (Python 3.8+)."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
METHODS = {
    "C32": "refit_compact32_active",
    "C64": "refit_compact64_active",
    "learned_moments": "refit_moment_learned",
}
SEEDS = [1701, 1702, 1703, 1704, 1705]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_archive(directory, sources):
    """Recover packaged observations from original results and complete-worker CPU records."""
    docs = {}
    for name, source in sources.items():
        path = directory / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Archive source failed checksum: {path}")
        docs[name] = json.loads(path.read_text(encoding="utf-8"))

    by_energy = {}
    for stratum in docs["forecast"]["strata"].values():
        models = stratum["methods"]
        losses = {}
        for name, key in METHODS.items():
            entries = models[key]
            if entries["seeds"] != SEEDS:
                raise ValueError("Unexpected controlled-fit seed order")
            losses[name] = entries["groups"]["all_actions"]["per_fit_response_raw_nmse"]
        passive = models["compact32_passive"]["groups"]["all_actions"]["metrics"]
        active = models["compact32_active"]["groups"]["all_actions"]["metrics"]
        physical = models["moment_physics"]["groups"]["all_actions"]["metrics"]
        by_energy[str(int(stratum["energy_eV"]))] = {
            "families": stratum["families"],
            "cases": stratum["cases"],
            "per_fit_raw_response_nmse": losses,
            "physical_moments_raw_response_nmse": physical["response_raw_nmse"]["mean"],
            "original_C32_active_raw_response_nmse": active["response_raw_nmse"]["mean"],
            "original_C32_passive_raw_response_nmse": passive["response_raw_nmse"]["mean"],
        }

    reports = docs["cpu"]["reports"]
    census_reports = docs["census_cpu"]["reports"]
    def timings(workload, records):
        return [records[f"repeat{i}/{workload}"]["fresh_process_CPU_seconds"] for i in range(3)]

    receipt = reports["repeat0/paired32/refit_moment_learned"]["output_receipt"]
    if receipt["seed"] != SEEDS[0] or not receipt["paired"]:
        raise ValueError("Unexpected timed model or workload")
    irt = docs["irt"]["strata"]["energy400"]["groups"]["all_actions"]["IRT_expected_MSE"]
    utility = docs["utility"]["strata"]["energy400"]["methods"]
    allocations = []
    for model, m, paths in [("native_only", 8, 8), ("native_only", 64, 1),
                            ("refit_moment_learned", 8, 8), ("native_only", 8, 64),
                            ("native_only", 64, 8), ("refit_moment_learned", 8, 64)]:
        fit = utility[model]["fits"][0]
        if model != "native_only" and fit["seed"] != SEEDS[0]:
            raise ValueError("Unexpected fit in the mean-yield comparison")
        match, = [row for row in fit["allocations"] if row["m"] == m and row["R"] == paths]
        allocations.append({"method": model, "sampled_sources": m, "paths_per_source": paths,
                            "estimated_MSE": match["finite_bank_MSE"]["mean"]})
    return {
        "seeds_in_array_order": SEEDS,
        "by_energy_eV": by_energy,
        "state_bytes_per_branch": {m: docs["forecast"]["strata"]["energy400"]["methods"][key]
                                   ["storage"]["state_bytes"] for m, key in METHODS.items()},
        "fitting_CPU_seconds_1701": docs["fit"]["elapsed_cpu_seconds"],
        "forecast_400eV": {
            "timed_paired_cases": receipt["cases"],
            "irt64_raw_response_nmse": irt["64"]["finite_reference"]["standardized_seven_species_MSE"]["mean"],
            "CPU_seconds_all_three_repeats": {
                "learned_moments_1701": timings("paired32/refit_moment_learned", reports),
                "IRT_64_paths_per_arm": timings("paired_IRT_R64", reports),
            },
        },
        "yield_400eV": {
            "allocations": allocations,
            "CPU_seconds_all_three_repeats": {
                "acquire_8": timings("source_SBS_m8", reports),
                "acquire_64": timings("source_SBS_N64", reports),
                "model_64": timings("passive_N64/refit_moment_learned", reports),
                "native_8x8": timings("native_SBS_R8", reports),
                "native_8x64": timings("native_SBS_R64", reports),
                "native_64x1": timings("native_SBS_census_R1", census_reports),
                "native_64x8": timings("native_SBS_census_R8", reports),
            },
        },
    }


def replay(data):
    seeds = data["seeds_in_array_order"]
    energy = data["by_energy_eV"]
    methods = energy["400"]["per_fit_raw_response_nmse"]
    moment = methods["learned_moments"]
    moment_mean = mean(moment)
    c32_exceptions = [
        {"moment_seed": mseed, "C32_seed": cseed} for mseed, merr in zip(seeds, moment)
        for cseed, cerr in zip(seeds, methods["C32"]) if merr >= cerr
    ]
    forecast = data["forecast_400eV"]
    times = {name: median(repeats) for name, repeats in forecast["CPU_seconds_all_three_repeats"].items()}
    first_fit = times["learned_moments_1701"]
    irt = times["IRT_64_paths_per_arm"]
    saving = (irt - first_fit) / forecast["timed_paired_cases"]
    utility = data["yield_400eV"]
    costs = utility["CPU_seconds_all_three_repeats"]
    tables = []
    for row in utility["allocations"]:
        m = row["sampled_sources"]
        r = row["paths_per_source"]
        assisted = row["method"] != "native_only"
        keys = [f"native_{m}x{r}", "acquire_64" if assisted else f"acquire_{m}"]
        if assisted:
            keys.append("model_64")
        # Add costs from the same repetition before taking the median.
        charged = median(sum(costs[key][i] for key in keys) for i in range(3))
        tables.append({**row, "continuations": m*r, "median_acquisition_charged_CPU_seconds": charged})
    return {
        "fixed_fit_mean_raw_response_nmse_by_eV": {
            e: {m: mean(values) for m, values in group["per_fit_raw_response_nmse"].items()}
            for e, group in energy.items()},
        "400eV_moments_percent_reduction_vs_C64": 100 * (1 - moment_mean / mean(methods["C64"])),
        "state_bytes_per_branch": data["state_bytes_per_branch"],
        "400eV_fit_ranges": {m: [min(v), max(v)] for m, v in methods.items()},
        "400eV_full_separation_moments_vs_C64": max(moment) < min(methods["C64"]),
        "400eV_equal_or_better_C32_pairs": c32_exceptions,
        "400eV_physical_moments_raw_response_nmse": energy["400"]["physical_moments_raw_response_nmse"],
        "original_C32_passive_over_active_NMSE": {
            e: energy[e]["original_C32_passive_raw_response_nmse"]/energy[e]["original_C32_active_raw_response_nmse"]
            for e in ["100", "400"]},
        "400eV_timed_fit_raw_response_nmse": moment[0],
        "400eV_irt64_raw_response_nmse": forecast["irt64_raw_response_nmse"],
        "400eV_median_forecast_CPU_seconds_per_batch": times,
        "400eV_forecast_CPU_ratio_IRT_over_timed_fit": irt / first_fit,
        "400eV_CPU_seconds_saved_per_paired_case": saving,
        "seed_1701_fit_CPU_hours": data["fitting_CPU_seconds_1701"] / 3600,
        "illustrative_cases_to_offset_fitting_CPU": data["fitting_CPU_seconds_1701"] / saving,
        "400eV_finite_bank_OH": tables,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icwm", type=Path, help="optionally verify inputs against the full archived icwm directory")
    args = parser.parse_args()
    bundle = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    if args.icwm and read_archive(args.icwm, bundle["sources"]) != bundle["data"]:
        raise ValueError("Supplied observations differ from verified archive sources")
    results = replay(bundle["data"])
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
