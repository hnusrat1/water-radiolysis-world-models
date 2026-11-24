Predicting Pulse Responses in Water Radiolysis with Compact World Models
Manuscript v13 | Exploratory transfer to new pulse schedules at 200 eV

The main paper evaluates source transfer at familiar pulse settings. This separate follow-up applies its fifteen frozen learned fits and unfitted physical moments to an earlier held-out 200 eV bank. The earlier first-order predictors had previously been evaluated on these families. The current fits used no target data from these families for training or model selection. The new evaluation plan was fixed before producing or scoring the current trajectories.

The 64 families each supplied nine actions. Six actions combined a 3 or 5 ns delay with a 0, 10 or 50 nm offset for simultaneous injection of three incoming clouds. Three actions supplied one of the three clouds at each of 1, 5 and 10 ns, with the same three offsets. Each action has 64 independent native paths per branch. The 576 cases have 73,728 archived native paths. Species order is hydrated electron, OH radical, hydrogen atom, hydrogen molecule, hydronium, hydroxide, and hydrogen peroxide.

For each fixed fit and each family, family_losses.json records the mean 20 ns raw response NMSE over the six or three actions in the corresponding stratum. At each action, the endpoint averages squared errors over the seven training-standardized species. All group scores average individual-fit losses. Whole families supply the paired bootstrap units. The same 4,000 family resamples are shared across fits, methods, and strata. The intervals are descriptive, condition on the observed five fits per learned group, and are unadjusted. plan.json gives the source and checkpoint identities, target label identity, and prespecified tolerances. scoring_freeze.json records the score and replay code hashes fixed before opening the new action scores.

Quick replay

From the extracted action-transfer folder, run the following using Python 3.12 and NumPy.

    python -B replay.py

The JSON result should match expected_results.json. No model weights or native trajectories are needed for this replay. verification.json gives the numerical and positive-control maxima. numerical_gate.json gives checks for each predictor.

Recompute model forecasts from the public weights and native reference data

Use the code and weights in the inference directory of the public repository:
https://github.com/hnusrat1/water-radiolysis-world-models/tree/v1.0.0/inference

Download ICWM_v13_ablation_training_and_evaluation_inputs.zip from release v1.0.0:
https://github.com/hnusrat1/water-radiolysis-world-models/releases/download/v1.0.0/ICWM_v13_ablation_training_and_evaluation_inputs.zip

Keep the input archive as a ZIP. From the repository root, with Python 3.12,
install the CPU inference dependencies and run

    python -m pip install -r inference/requirements.txt
    python supporting/action_transfer/verify_forecasts.py /path/to/ICWM_v13_ablation_training_and_evaluation_inputs.zip
    python supporting/action_transfer/verify_forecasts.py /path/to/ICWM_v13_ablation_training_and_evaluation_inputs.zip --full

The first command that runs verify_forecasts.py reconstructs a three-pulse
schedule using the original first batch. It compares all eleven query times,
both branch counts and the two water ledgers with the archived results for
case 2016012 and learned-moment seed 1701. The full run forecasts all 576
cases with the fifteen published fits and unfitted physical moments. It reads
the 64 native paths per branch, checks their stored means, reconstructs
all 2,048 family losses across the two schedule groups, and checks the paired
intervals reported in expected_results.json. Both modes use a single CPU
thread. To retain the verification report, add --json report.json.
The archived CPU run and its file identities are in public_replay_verification.json.

The same program can run directly from this supplementary folder. Pass
--inference /path/to/water-radiolysis-world-models/inference to locate the
downloaded repository. The original analysis plan, scoring and numerical
gate receipts document the original reference and refinement tests.
