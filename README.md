# AI651 Assignment 1

**Burhan Ahmad — 27100405**

This repository contains the completed Task 1 notebook, the Task 2 Autoformer
implementation and saved experiment results, and the combined report.

## Submission files

- [Assignment1_Report.pdf](Assignment1_Report.pdf): report covering both tasks.
- [Assignment1_Report.tex](Assignment1_Report.tex): complete LaTeX source, including both tasks and AI-use disclosure.
- [Question 1/Assignment1.ipynb](Question%201/Assignment1.ipynb): executed full-preset notebook with all three exercises implemented, checks passed, numbered outputs, and deployment choices.
- `Question 1/results/design/`: original numbered PDF figures and CSV result tables.
- `Question 2 - Leaderboard/AI651_Task2_Autoformer_Kaggle.ipynb`: source used for the completed Kaggle experiment.
- `Question 2 - Leaderboard/results/`: screening, final fold/seed metrics, horizon errors, configuration, provenance, and derived audit tables.
- `Question 2 - Leaderboard/figures/`: original EDA/forecast plots and the shared-axis horizon comparison.
- `Question 2 - Leaderboard/submission/`: original prediction CSV, the 168-value leaderboard paste list, and parameter/epoch metadata.

The optional Task 2 reflection questions are omitted. The public repository URL
appears on the report's first page.

## Task 2 results

The saved experiment is version **2** of
`burhanahmadkhanbak/ai651-task2-autoformer-kaggle`. Final evaluation uses seeds
2026 and 3407 at chronological origins 43152, 43320, and 43488. These are six
model-selection evaluations, not hidden-test results; RMSE spread below is the
sample standard deviation of the six run-level RMSEs.

| Autoformer variant | RMSE mean ± SD | MAE | sMAPE (%) | Parameters |
| --- | ---: | ---: | ---: | ---: |
| With external A–J | 58.72 ± 16.23 | 45.25 | 82.61 | 262,849 |
| Without external data | 88.41 ± 23.55 | 76.54 | 102.41 | 260,929 |

External inputs improve RMSE in all six paired runs, reducing mean RMSE by
33.58%. The selected configuration uses raw target standardization, context
336, label length 84, width 96, four heads, two encoder layers, one decoder
layer, feed-forward width 192, moving-average kernel 25, factor 3, dropout 0.1,
AdamW learning rate/weight decay 1e-4, and batch size 32.

The final model was freshly trained on all 43,656 observed targets for three
epochs. Its 168 saved forecasts cover indices 43657–43824. The leaderboard
expects the comma-separated values in `submission/leaderboard_predictions.txt`
and parameter/epoch declarations, rather than a CSV upload. Parameter count is
**262849**. Metadata distinguishes the **3 full-refit epochs**, **30 selected-variant
final-CV epochs**, and **117 complete-pipeline epochs**; it uses the latter as a
conservative declaration including all screening and ablation training.
`results/epoch_accounting.csv` documents how early-stop run lengths were derived.
No leaderboard submission was made during repository preparation.

The original notebook leaves dropout active during validation and prediction.
Saved metrics and forecasts retain that protocol, which is disclosed in the
report. The same chronological folds were used for checkpoint and architecture
selection. These limitations qualify the results; no training was rerun to
replace them.

## Reproduce or inspect

Use Python 3.11 or later and install the requirements for the task you wish to
run. For Task 1, start Jupyter in `Question 1/` and set `PA1_PRESET=full` before
running. For Task 2, start from this root or its notebook directory; the supplied
CSV data are included under `Question 2 - Leaderboard/Data/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 'Question 1/requirements.txt' -r 'Question 2 - Leaderboard/requirements.txt'
jupyter lab
```

Full training takes hours. To inspect and audit the saved results without
training, run:

```bash
python 'Question 2 - Leaderboard/prepare_submission.py'
```

To compile the report, run `tectonic Assignment1_Report.tex` or
`latexmk -pdf Assignment1_Report.tex` from this root. Keep the referenced figure
directories alongside the source. A local LaTeX compiler is required for this
report's external images; the Codex standalone compiler does not package them.

The optional `download_completed_run.py` retrieves saved Kaggle version 2 using
owner authentication (`kaggle>=2.2,<3`). Checkpoints, raw download duplicates,
local environments, credentials, and obsolete report copies are excluded from
Git. Source notebooks, data, the report, lightweight results, and required
figures are tracked.
