# Transfer endpoints at 20 ns

The two NumPy archives contain the observations used to score pulse response at 100 and 400 eV per primary electron. Each energy contributes 64 fresh source families. The same four pulse conditions were applied to each family, giving 256 cases per energy. Each case has independent step-by-step (SBS) reference simulations for the pulse and passive branches, with 64 continuations per branch. Both files together contain 65,536 seven-species simulator endpoints.

## Arrays in each file

| Field | Shape | Interpretation |
| --- | --- | --- |
| `energy_eV`, `time_ns` | scalar each | Energy per primary electron and time of observation in nanoseconds |
| `case_id`, `family_id` | 256 each | Original case and source-family identifiers, aligned across all arrays |
| `pulse_time_ns`, `spacing_nm` | 256 each | Time of the pulse (1 or 10 ns) and distance *r* (0 or 50 nm). The three incoming clouds arrive together at x offsets −*r*, 0 and +*r*. |
| `species` | 7 | Hydrated electron (`e_aq`), hydroxyl radical (`OH`), hydrogen atom (`H`), molecular hydrogen, hydronium, hydroxide and hydrogen peroxide |
| `scale` | 7 | Root-mean-square passive training count for each species, with a minimum of one molecule |
| `seeds` | 5 | Fixed fit seeds in order, 1701 through 1705 |
| `sbs_forced`, `sbs_passive` | 256 × 64 × 7 | Integer counts in molecules, indexed by case, continuation and species |
| `moments_forced`, `moments_passive`, `C32_forced`, `C32_passive`, `C64_forced`, `C64_passive` | 5 × 256 × 7 | Predicted mean counts in molecules, indexed by fit, case and species |

## How the scores are formed

For each case and species, average the 64 SBS continuations for each branch, then subtract the passive mean from the pulse mean. For each fit, compare its predicted pulse-minus-passive count with that reference. Square the discrepancy, divide by the squared species scale and average over the seven species. The raw normalized mean squared error (NMSE) includes the sampling error in the 64-path reference. The family loss gives equal weight to the five fits and the four actions. The primary contrast subtracts C64 loss from learned-moment loss within each family, then gives equal weight to the two energy strata. [Endpoint scoring code](../score_endpoints.py)

## Provenance

The two NPZs preserve the 20 ns slice of the archived trajectories at the original numerical precision. During extraction, the case and family IDs were matched across simulation labels and predictions. The mean of every extracted set of 64 SBS counts was also checked against the corresponding archived branch mean, for all cases and species. The SHA-256 values identify the seven upstream files used for extraction. [Upstream file hashes](upstream_sha256.json)

The inputs were drawn from the six-stage companion archive `radiochemistry_reviewer_reproduction_v1.zip` (SHA-256 `7112d72245e4fbaae2cfd115eaee0141f0ccbd77fddf353c212adf4ab4a42f23`). The repository includes both endpoint files needed for the analysis. A reader with a copy of the larger archive can regenerate them from the repository root with the following command.

```sh
python primary/extract_endpoints.py /path/to/extracted/archive/data/icwm/extension/strengthening_v1 rebuilt_endpoints
```
