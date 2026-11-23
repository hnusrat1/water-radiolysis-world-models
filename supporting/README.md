# Supporting analyses and figures

Run the commands below from the repository root. All inputs for these analyses are included. The [main README](../README.md) gives the primary paired comparison over the 100 and 400 eV source families.

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

## Redraw the paper's figures

With the dependencies in the root requirements-figures.txt installed, run the two commands in order. The first creates the rendered directory.

```bash
python supporting/figure_replay/render.py --output-dir rendered
python supporting/training_diagnostic/plot_fit_trajectories.py supporting/training_diagnostic/fit_trajectories.csv rendered/fit_trajectories.pdf
```

The first command creates Figures 1–4 from the [archived plotting inputs](figure_replay/plot_inputs.json). They cover the source cloud and intervention, transfer across energy, a representative forecast, and conditional forecast cost. The second creates the appendix figure of 200 eV training and validation trajectories from [367 logged observations](training_diagnostic/fit_trajectories.csv) across 15 fits. The open circles mark the checkpoints chosen by validation. [Fit diagnostics](training_diagnostic/fit_diagnostic_evidence.json) include each selected epoch and validation score, the held-out errors and the digests of the underlying records.

## Provenance

Every data file and Python script in this directory is a byte-for-byte copy of the corresponding member of the manuscript supplement ICWM_supplement_v9.zip (SHA-256 `9c566a97623c30022717282ba492c5d3e6f7185f935de95af3bc2e1e08b5bcaf`). The numerical environment used Python 3.12.15 and NumPy 2.5.4. The published figure rendering used Matplotlib 3.11.2, bundled DejaVu Sans and FreeType 2.14.3. A different plotting environment can produce different PDF bytes.
