# AI651 Assignment 1

Burhan Ahmad — 27100405

The combined report is **Assignment1_Report.pdf**. Its main LaTeX source is
**Assignment1_Report.tex**, which includes **Task2_Report_Section.tex**. Compile
from the repository root with a normal LaTeX distribution, for example:

```bash
latexmk -pdf Assignment1_Report.tex
# or
tectonic Assignment1_Report.tex
```

The report retains the existing five Task 1 responses and adds the completed
Task 2 experiment. The optional Q2.1–Q2.9 reflection component is omitted.
The source references the original numbered Task 1 PDF figures and Task 2
figures by their repository paths; keep those directories with the source.

## Task 2 implementation and results

The authoritative implementation is
`Question 2 - Leaderboard/AI651_Task2_Autoformer_Kaggle.ipynb`.
It implements progressive moving-average decomposition, FFT autocorrelation,
time-delay aggregation, and an encoder–decoder with known-future external inputs.
The notebook lives in the Task 2 directory in this checkout.

The completed saved Kaggle run is version **2** of
`burhanahmadkhanbak/ai651-task2-autoformer-kaggle`.
Lightweight original tables and the run manifest are in
`Question 2 - Leaderboard/results/`; provenance records hashes of the original
artifacts and input data. The local notebook's sources match the current remote
notebook sources, and its architecture matches the saved version 2 manifest and
checkpoint. The downloaded bundle did not contain an executed notebook.

Final comparison uses seeds 2026 and 3407 at origins 43152, 43320, and 43488.
Reported means below are arithmetic means over the six fold/seed metrics;
RMSE spread is sample standard deviation across those six runs.

| Variant | RMSE mean ± SD | MAE | sMAPE (%) | Parameters |
| --- | ---: | ---: | ---: | ---: |
| With A–J | 58.72 ± 16.23 | 45.25 | 82.61 | 262,849 |
| Without external data | 88.41 ± 23.55 | 76.54 | 102.41 | 260,929 |

The with-external model improves RMSE in all six matched pairs and reduces its
mean by 33.58%. Selected settings are raw global target standardization,
context 336, decoder label length 84, model width 96, four heads, two encoder
layers, one decoder layer, feed-forward width 192, moving average 25, factor 3,
dropout 0.1, AdamW learning rate/weight decay 1e-4, and batch size 32.

The final model was initialized afresh with seed 2026, fitted on all 43,656
observed targets for **three epochs**, and produced the original saved 168
forecasts. No training was rerun during repository preparation.

## Leaderboard preparation

The assignment requires pasting **168 comma-separated numbers**, not uploading
a CSV. Prepared files are under `Question 2 - Leaderboard/submission/`:

- `submission.csv`: the unmodified saved Kaggle forecast, indices 43657–43824.
- `leaderboard_predictions.txt`: exactly 168 values in chronological order.
- `leaderboard_metadata.json`: parameter count and explicit epoch accounting.

**P = 262849.** The manifest's **3 epochs** describes only the full-history fit.
The handout also says to count preceding early-stopping training alongside a
refit. `results/epoch_accounting.csv` records the original pipeline: one smoke
epoch, 55 screening epochs, 30 with-external final-CV epochs, 28 without-external
final-CV epochs, and three full-fit epochs, totaling **117**. Prepared metadata
uses **E = 117 as a conservative complete-pipeline declaration**, explicitly
including selection/ablation training rather than equating E with the refit
count. Early-stopping run lengths are derived from saved best epochs and the
original patience/cap logic; they are not newly measured logs.

No leaderboard submission was made during preparation. No hidden test score is
reported, and leaderboard feedback was not used for model selection.

## Reproduction

Use Python 3.11 or later. Install dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 'Question 2 - Leaderboard/requirements.txt'
jupyter lab
```

Open the Task 2 notebook and run top to bottom to reproduce the original search
and final training. It searches for the three supplied CSVs; start Jupyter from
this root or the Task 2 directory. On Kaggle, attach the dataset containing
those CSVs. Full training is expensive; it is unnecessary for inspecting the
committed report and tables.

To audit saved artifacts and regenerate the shared-axis horizon figure and
paste list **without training**:

```bash
python 'Question 2 - Leaderboard/prepare_submission.py'
```

To retrieve the private original run as its owner, install `kaggle>=2.2,<3`,
authenticate locally with `kaggle auth login`, and run:

```bash
python 'Question 2 - Leaderboard/download_completed_run.py'
```

This downloads version 2, not the currently running live session. Kaggle CLI
2.2.4's `kernels output` returned empty live-session output during preparation,
so the helper uses the official authenticated SDK's saved-version endpoint.
The raw download and model checkpoint remain local and are ignored by Git.

## Evidence and limitations

The original notebook leaves models in training mode in `fold_pred` and
`predict_test`, so dropout remains active even under `no_grad`. Tables and
forecasts document that original stochastic inference protocol. This issue is
disclosed in the report; the notebook, metrics, and predictions were not silently
replaced. Future deterministic evaluation should call `model.eval()` and restore
training mode before training resumes; its metrics must be measured anew.

The same recent horizons were used for early stopping and architecture selection.
Six fold/seed runs therefore provide selection evidence, not an independent
final-test estimate or a confidence interval. All external features were ablated
jointly; individual-feature and historical-versus-future contributions were not
isolated. No single fixed operating period is assumed.

**Task 1 submission item still needs reconciliation:** although Task 1's report
and result exports are present, the current local `Question 1/Assignment1.ipynb`
contains implementation placeholders and no saved execution outputs. It was
preserved as instructed, not rerun or rewritten. Replace it with the already
completed, executed full-preset notebook before submitting the assignment.
`harness/` and `requirements.txt` remain beside it. The handout requires that
executed notebook in addition to the combined PDF, its LaTeX source, and each
numbered PDF figure included in the report.

AI assistance is disclosed at the end of the report. Earlier prompts behind the
previously supplied Task 1 draft were not available for reconstruction.
