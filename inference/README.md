# Recompute the transfer forecasts from molecular coordinates

These fifteen checkpoints accompany *Predicting Pulse Responses in Water Radiolysis with Compact World Models* by Humza Nusrat and Omar Nusrat. The verifier computes the 20 ns counts for seven species on both branches of each case. It then compares the branch counts and their difference with the archived forecasts used in the paper. [Manuscript citation](../CITATION.cff)

## Run on CPU

From the repository root, install the tested dependencies in your Python environment. The release was tested with Python 3.12.15.

```sh
python -m pip install -r inference/requirements.txt
python inference/verify.py
```

The quick check runs the first 128 cases at 400 eV through the learned-moment fit with seed 1701. To evaluate the 512 cases at 100 and 400 eV with all five fits of each of the three models, run the full check.

```sh
python inference/verify.py --full
```

Each comparison uses an absolute tolerance of 0.00000001 molecules and zero relative tolerance. A successful full run reports 7,680 paired forecasts, the largest absolute differences in branch counts and pulse responses, and elapsed time. On an arm64 Mac running macOS 26.2, the full replay took 566.2 seconds on one CPU thread. The maximum absolute differences in branch counts and pulse responses were both zero molecules. The [primary replay](../README.md#reproduce-the-primary-result) uses the archived forecasts and individual simulation outcomes to compute normalized errors and their paired confidence intervals.

## Inputs and evaluation

The compressed JSON files in [cases](cases/) hold 256 cases at each energy, in the paper’s original order. Each entry supplies the initial particles, incoming particles, start time, pulse time, query times and identifiers. A particle is a species ID followed by three coordinates in nanometers. Each source family contributes four pulse conditions. The study used 64 separately simulated outcomes per branch to estimate the reference response.

The code loads the original fitted parameters from the tensor-only files in [weights](weights/). It uses the study’s adaptive SSPRK3 solver in float64, the original input rounding through float32, and the same batches of 128 and eleven query times. It runs the two branches separately using deterministic operations on one CPU thread. The [model configuration](config.json) gives the solver tolerances, validation-selected checkpoints, and hashes for the original and exported weights. The [provenance record](provenance.json) identifies the inputs and code.

The reaction and diffusion model uses the Geant4-DNA chemistry described in the paper. The CPU replay runs from supplied coordinates and fitted weights. See the [third-party notices](THIRD_PARTY_NOTICES.txt) and [Geant4 license](Geant4_LICENSE.txt).
