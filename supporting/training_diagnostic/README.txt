Predicting Pulse Responses in Water Radiolysis with Compact World Models
Manuscript v13 | Training and validation trajectories at 200 eV

fit_trajectories.csv contains the original epoch records for C32, C64 and learned
moments, fitted at seeds 1701 through 1705. Each of the 367 rows gives method,
seed, epoch, training loss, validation score, and a binary checkpoint-selection
indicator. Epoch zero has a validation score and an empty training-loss field.
The trainers logged every fifth epoch. Fourteen fits reached epoch 120. C32 seed
1701 stopped at epoch 80 after eight checks without an improvement, retaining
the checkpoint selected at epoch 40.

The training loss is the mean over six 128-case batches during the logged
epoch. Each batch contributes its loss immediately before its optimizer update.
The objective gives unit weight to the trajectory error, averaged over the
forced and passive branches, and half weight to the terminal response error.
The trajectory term includes the initial query. The validation score uses the
weights at the end of the epoch and the complete held-out 200 eV validation
set. It gives unit weight to terminal response error and weight 0.1 to the
branch-averaged trajectory error over post-initial queries. Both metrics use
the frozen species scales. Their weights and evaluation procedures differ.
Read trends within each row of the figure. The open circles mark the first
strict minimum in validation score for each fit.

fit_diagnostic_evidence.json records the per-fit selected scores and epochs,
the 400 eV confirmation errors, and source digests. The original audit checked all
15 report and checkpoint hashes and replayed the strict-improvement and
early-stopping rules. For C64 seed 1705 it also cross-checked the two worker
logs on either side of its restart. The public training_records files
reproduce the curves and checkpoint choices. The evidence file also records the
original checkpoint, fit-record, protocol and worker-log digests.

The evidence file gives the all-action raw response errors of each
learned-moment fit on the 100, 200 and 400 eV confirmation banks. It also
compares the 400 eV error of learned-moment seed 1702
with the mean error of the other four seeds for each action. The four action
groups have equal counts. The fraction attributed to each action is its
excess error divided by the sum of the four excesses. This describes where
the error accumulated. Seed 1702 had the highest error for all four actions.

To draw the figure from the supplied CSV, run this command in the directory
containing the diagnostic files with Python 3.10+ and Matplotlib.

  python3 plot_fit_trajectories.py fit_trajectories.csv fit_trajectories.pdf

To rebuild and check the CSV from the original 15 fit records, run

  python3 rebuild_history.py --output rebuilt_trajectories.csv

The training_records directory contains each validation log with its original
SHA-256 identity. This standard-library replay checks the selected checkpoint
and the complete fit roster, then compares the regenerated CSV byte for byte
with the file accompanying the plot. fit_diagnostic_evidence.json records the
400 eV comparisons against the five selected fits.

SHA-256 of fit_trajectories.csv
40bfe26864b481e9b6458cb8437e8b4878c14077c77a97ae33ebbf33aa93aeaf
