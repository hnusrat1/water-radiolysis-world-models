# Supporting analyses and figures

Run the commands below from the repository root. The compact replays use files included here. The temporal inference check also uses the linked data download. The [main README](../README.md) gives the primary paired comparison over the 100 and 400 eV source families.

## Fit-level and computational results

```bash
python supporting/headline_replay/replay.py
```

This standard-library script prints JSON with fixed-fit means and ranges, the original passive-versus-active C32 comparison, conditional forecast timings, and mean hydroxyl-radical yield estimation costs. For the 400 eV transfer test it reports mean raw response NMSE of 0.080127 for learned moments, 0.137892 for C32 and 0.185802 for C64. All five learned-moment fits scored below all five C64 fits. For the preselected fit, the 32-case conditional forecast used 3.94 CPU-seconds compared with 74.92 for 64-path independent reaction time simulation. The timing calculation uses the recorded CPU measurements.

[Fit and timing inputs](headline_replay/inputs.json) include each fitted seed, timing repetitions, and SHA-256 digests for six underlying study reports. The [per-species table](headline_replay/species_response_400eV.csv) gives raw response RMSE in molecules, averaged in squared-error space over five fixed fits, 64 fresh 400 eV source families and four actions. OH denotes the hydroxyl radical and OH- denotes hydroxide.

## Transfer to new pulse schedules

```bash
python supporting/action_transfer/replay.py
```

This NumPy script uses [per-family losses](action_transfer/family_losses.json) from 64 held-out 200 eV families to score two exploratory tests. One changed the delay for a single addition. The other supplied three successive additions at 1, 5 and 10 ns. For the three-pulse schedule the mean raw NMSE was 0.02552 for learned moments and 0.07538 for C64. Their paired difference was −0.04986 with a descriptive 95% interval of −0.05701 to −0.04291. The bootstrap resamples whole families and conditions on the fitted predictors.

[Expected output](action_transfer/expected_results.json) contains the full-precision estimates. The [analysis plan](action_transfer/plan.json) records the checkpoint, simulator-label and engine hashes. The [scoring record](action_transfer/scoring_freeze.json), [numerical checks](action_transfer/numerical_gate.json), [verification record](action_transfer/verification.json) and [independent checks](action_transfer/independent_verification.json) document the checks on those inputs.

To check the result from molecular coordinates, download the [input archive](https://github.com/hnusrat1/water-radiolysis-world-models/releases/download/v1.0.0/ICWM_v13_ablation_training_and_evaluation_inputs.zip). In the repository's CPU inference environment, run the following with the path to the downloaded ZIP.

```sh
python supporting/action_transfer/verify_forecasts.py /path/to/ICWM_v13_ablation_training_and_evaluation_inputs.zip --full
```

The full check reconstructs all 9,216 paired forecasts and compares all family losses and paired intervals against the published values. Omit `--full` to check one three-pulse case with the preselected moment fit. [Temporal forecast instructions](action_transfer/README.txt) · [Verified CPU run](action_transfer/public_replay_verification.json)

## Redraw the paper's figures

With the dependencies in the root requirements-figures.txt installed, run the two commands in order. The first creates the rendered directory.

```bash
python supporting/figure_replay/render.py --output-dir rendered
python supporting/training_diagnostic/plot_fit_trajectories.py supporting/training_diagnostic/fit_trajectories.csv rendered/fit_trajectories.pdf
```

The first command creates Figures 1–4 from the [archived plotting inputs](figure_replay/plot_inputs.json). They cover the source cloud and intervention, transfer across energy, a representative forecast, and conditional forecast cost. The second creates the appendix figure of 200 eV training and validation trajectories from [367 logged observations](training_diagnostic/fit_trajectories.csv) across 15 fits. The open circles mark the checkpoints chosen by validation. [Fit diagnostics](training_diagnostic/fit_diagnostic_evidence.json) include each selected epoch and validation score, the held-out errors and the digests of the underlying records.

To verify the training curves directly from the [15 original fit records](training_diagnostic/training_records), run the following. It checks each record's identity and the selected checkpoint before reproducing the CSV byte for byte.

```sh
python supporting/training_diagnostic/rebuild_history.py
```

## Mean-yield error and CPU time

To regenerate the appendix figure comparing all 14 estimator and path allocations, use the same plotting environment as the main figures.

```bash
python supporting/figure_replay/utility_figure.py utility_charged_cost_B400.pdf
```

The horizontal axis includes CPU time for generating the sources. The 64-path and 512-path panels each compare an eight-source correction with direct sampling across all 64 sources.

## Provenance

The archived primary fit and endpoint records are fixed. The updated figure scripts use the same numerical observations as the v13 manuscript. The numerical environment used Python 3.12.15 and NumPy 2.5.4. The vector figures were drawn with Matplotlib 3.11.2, its bundled DejaVu Sans and FreeType 2.14.3. The versioned release includes the input digests, standalone computational checks and recorded outputs. [Code and data release](https://github.com/hnusrat1/water-radiolysis-world-models/releases/tag/v1.0.0)
