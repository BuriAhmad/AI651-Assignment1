"""Audit completed-run artifacts and prepare report evidence; never train a model.

Run from any directory. Requires numpy, pandas, and matplotlib. Original
tables are retained unchanged; derived summaries are explicitly identified.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SUBMISSION = ROOT / 'submission'
FIGURES = ROOT / 'figures'


def main():
    manifest = json.loads((RESULTS / 'run_manifest.json').read_text())
    cv = pd.read_csv(RESULTS / 'final_cv_results.csv')
    summary = pd.read_csv(RESULTS / 'final_cv_summary.csv').set_index('experiment')
    screen = pd.read_csv(RESULTS / 'screening_results.csv')
    h = pd.read_csv(RESULTS / 'final_horizon_rmse.csv')
    submission = pd.read_csv(SUBMISSION / 'submission.csv')
    train = pd.read_csv(ROOT / 'Data/student_train.csv')
    test = pd.read_csv(ROOT / 'Data/student_test.csv')
    ext = pd.read_csv(ROOT / 'Data/optional_external_data.csv')
    assert train.shape == (43656, 2) and test.shape == (168, 2)
    assert list(train.columns) == list(test.columns) == ['time_idx', 'value']
    assert np.array_equal(train.time_idx, np.arange(1, 43657))
    assert np.array_equal(test.time_idx, np.arange(43657, 43825))
    assert train.value.notna().all() and (train.value >= 0).all() and test.value.isna().all()
    assert ext.shape == (43824, 11) and np.array_equal(ext.time_idx, np.arange(1, 43825))
    assert not ext.isna().any().any()
    binary = ['feature_G', 'feature_H', 'feature_I', 'feature_J']
    assert ext[binary].isin([0, 1]).all().all() and ext[binary].sum(axis=1).eq(1).all()
    assert submission.shape == (168, 2)
    assert list(submission.columns) == ['time_idx', 'value']
    assert np.array_equal(submission.time_idx, np.arange(43657, 43825))
    assert np.isfinite(submission.value).all() and (submission.value >= 0).all()
    assert manifest['selected_variant'] == 'FINAL_with_external'
    assert manifest['parameters'] == 262849 and manifest['final_epochs'] == 3
    for name, rows in cv.groupby('experiment'):
        assert len(rows) == 6 and set(rows.seed) == {2026, 3407}
        assert set(rows.origin) == {43152, 43320, 43488}
        assert not rows.duplicated(['origin', 'seed']).any()
        assert np.allclose(rows.rmse.mean(), summary.loc[name, 'mean_rmse'])
        assert np.allclose(rows.rmse.std(ddof=1), summary.loc[name, 'std_rmse'])
        assert np.allclose(rows.mae.mean(), summary.loc[name, 'mean_mae'])
        assert np.allclose(rows.smape.mean(), summary.loc[name, 'mean_smape'])
        for encoded in rows.config_json:
            cfg = json.loads(encoded)
            base = dict(manifest['selected_experiment'])
            base['exog_mode'] = 'all' if name.endswith('with_external') else 'none'
            assert cfg == base
    with_ext = cv[cv.experiment == 'FINAL_with_external'].set_index(['origin', 'seed'])
    without_ext = cv[cv.experiment == 'FINAL_without_external'].set_index(['origin', 'seed'])
    paired = with_ext[['rmse', 'mae', 'smape', 'best_epoch']].rename(
        columns={x: x + '_with' for x in ['rmse', 'mae', 'smape', 'best_epoch']})
    paired = paired.join(without_ext[['rmse', 'mae', 'smape', 'best_epoch']].rename(
        columns={x: x + '_without' for x in ['rmse', 'mae', 'smape', 'best_epoch']}))
    paired['rmse_reduction_pct'] = 100 * (1 - paired.rmse_with / paired.rmse_without)
    paired.reset_index().to_csv(RESULTS / 'paired_ablation.csv', index=False)

    # Complete early-stopping counts follow exactly from best_epoch, patience,
    # and the notebook's maximum: termination is min(best_epoch + patience, cap).
    ledger = []
    for stage, frame, cap, patience in [('screening', screen, 4, 1), ('final_cv', cv, 10, 2)]:
        for row in frame.itertuples():
            ledger.append({'stage': stage, 'experiment': row.experiment, 'origin': row.origin,
                           'seed': row.seed, 'best_epoch': row.best_epoch,
                           'epochs_executed': min(cap, int(row.best_epoch) + patience),
                           'basis': 'derived from original early-stop loop'})
    ledger.extend([
        {'stage': 'smoke', 'experiment': 'SMOKE_ONLY', 'origin': 43488, 'seed': 2026,
         'best_epoch': 1, 'epochs_executed': 1, 'basis': 'one-epoch notebook smoke run'},
        {'stage': 'full_refit', 'experiment': manifest['selected_variant'], 'origin': 43656,
         'seed': manifest['final_seed'], 'best_epoch': 3, 'epochs_executed': 3,
         'basis': 'saved checkpoint and run manifest'}])
    ledger = pd.DataFrame(ledger)
    ledger.to_csv(RESULTS / 'epoch_accounting.csv', index=False)
    assert ledger.epochs_executed.sum() == 117
    (SUBMISSION / 'leaderboard_predictions.txt').write_text(
        ', '.join(format(float(v), '.10g') for v in submission.value) + '\n')
    metadata = {'parameters': manifest['parameters'], 'epochs_full_refit': 3,
                'epochs_selected_variant_final_cv': 30,
                'epochs_complete_pipeline': 117,
                'conservative_declared_epochs': 117,
                'epoch_scope': 'Smoke, screening, both final ablation variants, and full refit; '
                               'conservative complete-pipeline count, including early-stopping epochs.',
                'prediction_count': 168, 'first_time_idx': 43657, 'last_time_idx': 43824,
                'ensemble': False, 'source': 'Unmodified saved Kaggle version 2 forecasts',
                'inference_mode': 'Original notebook left dropout active; no replacement predictions generated.'}
    (SUBMISSION / 'leaderboard_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    y = train.value.to_numpy(float)
    center = y - y.mean()
    den = np.dot(center, center)
    lags = [1, 2, 5, 12, 24, 48, 72, 96, 168, 336]
    acf = {str(lag): float(np.dot(center[:-lag], center[lag:]) / den) for lag in lags}
    raw = pd.Series(y)
    log = pd.Series(np.log1p(y))
    eda = {'basis': 'Recomputed from provided data using diagnostics present in notebook cells 5-7',
           'n': len(y), 'min': float(y.min()), 'max': float(y.max()),
           'mean': float(y.mean()), 'median': float(np.median(y)), 'std_population': float(y.std()),
           'skew_unbiased': float(raw.skew()), 'excess_kurtosis_unbiased': float(raw.kurt()),
           'quantiles': {str(q): float(np.quantile(y, q)) for q in [.90, .95, .99, .999]},
           'rolling_mean_std_corr_raw': float(raw.rolling(168).mean().corr(raw.rolling(168).std())),
           'rolling_mean_std_corr_log1p': float(log.rolling(168).mean().corr(log.rolling(168).std())),
           'global_acf': acf,
           'history_binary_fractions': ext.iloc[:43656][binary].mean().to_dict(),
           'future_binary_fractions': ext.iloc[43656:][binary].mean().to_dict()}
    local = []
    for start in range(0, len(y) - 672 + 1, 672):
        z = y[start:start+672]; z = z - z.mean()
        local.append(float(np.dot(z[:-24], z[24:]) / np.dot(z, z)))
    eda['local_lag24_range'] = [min(local), max(local)]
    joined = ext.iloc[:len(y)].drop(columns='time_idx').copy(); joined['target'] = y
    eda['contemporaneous_correlations'] = joined.corr().target.drop('target').to_dict()
    (RESULTS / 'eda_summary.json').write_text(json.dumps(eda, indent=2) + '\n')

    # One shared-axis figure is derived from the existing horizon-error table.
    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    for name, color, label in [('FINAL_with_external', '#007c91', 'With external data'),
                               ('FINAL_without_external', '#d1495b', 'Without external data')]:
        rows = h[h.experiment == name].sort_values('horizon_step')
        assert np.array_equal(rows.horizon_step, np.arange(1, 169))
        ax.plot(rows.horizon_step, rows.rmse, color=color, label=label)
    ax.set(xlabel='Forecast horizon (steps)', ylabel='RMSE (original target units)',
           title='Final evaluation: six fold/seed squared errors pooled at each step')
    ax.legend(); ax.grid(alpha=.18)
    fig.savefig(FIGURES / 'horizon_comparison.pdf'); plt.close(fig)
    audit = {'checks': 'Passed data continuity, target visibility, one-hot states, 168 ordered finite '
                       'nonnegative predictions, all 12 fold/seed rows, summary aggregation, '
                       'configuration consistency, epoch ledger, and horizon coverage.',
             'rmse_reduction_pct': float(100 * (1 - summary.loc['FINAL_with_external', 'mean_rmse'] /
                                                   summary.loc['FINAL_without_external', 'mean_rmse'])),
             'all_six_paired_rmse_improve': bool((paired.rmse_with < paired.rmse_without).all()),
             'task1_notebook_status': 'Local notebook remains unexecuted with placeholders; preserved.',
             'task2_code_limitation': 'train_fold/fold_pred/predict_test never call model.eval(); '
                                      'validation and final inference use active dropout.',
             'no_training_or_leaderboard_submission_performed': True}
    (RESULTS / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
