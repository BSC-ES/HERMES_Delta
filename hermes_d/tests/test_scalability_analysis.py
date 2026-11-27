"""Delta scalability categories, historical logs, run filtering and CLI outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hermes_d.utilities.scalability_analysis import ScalabilityAnalysis, get_arguments, run


def make_log(root, processes=2, replication=0, scale=1., current=True):
    rows = {('Init', 'TOTAL'): [10.] * processes, ('Calcul', 'TOTAL'): [40.] * processes,
            ('Write', 'TOTAL'): [20.] * processes, ('HERMES', 'TOTAL'): [75.] * processes,
            ('Init', 'Grid'): [2.] * processes, ('Init', 'Clip'): [1.] * processes,
            ('SNAP01', 'Init Horizontal'): [3.] * processes,
            ('SNAP01', 'Speciation'): [8.] + [1.] * (processes - 1),
            ('SNAP02', 'Speciation'): [1.] + [8.] * (processes - 1),
            ('SNAP01', 'Temporal'): [1.] * processes,
            ('SNAP01', 'Horizontal'): [2.] * processes}
    if current:
        rows.update({('Write', 'prepare_output'): [2.] * processes,
                     ('Write', 'write_time_step_emissions'): [15.] * processes,
                     ('Write', 'unit_conversion'): [1.] * processes,
                     ('Write', 'communication'): [2.] * processes,
                     ('Write', 'netcdf_write'): [10.] * processes,
                     ('Write', 'netcdf_write_unaccounted'): [1.] * processes,
                     ('Write', 'close_output'): [3.] * processes})
    frame = pd.DataFrame.from_dict(rows, orient='index', columns=[str(rank) for rank in range(processes)])
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=['section', 'subsection'])
    frame *= scale
    # Deliberately inaccurate old summary columns must not influence rank-based statistics.
    for stat in ('min', 'mean', 'max'):
        frame[stat] = -999.
    folder = root / str(replication)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'times_{processes:04d}.csv'
    frame.to_csv(path)
    return path


def analysis(root, **kwargs):
    return ScalabilityAnalysis(str(root / '<replication>/times_<processes>.csv'), root / 'output', **kwargs)


def test_cli_defaults_match_gr():
    args = get_arguments(['--log-pattern', 'times_<processes>.csv', '--out-dir', 'out'])
    assert args.timing_stat == 'max'
    assert args.timing_schema == 'auto'
    assert args.summary_style == 'auto'
    assert args.aggregation_method == 'mean'
    assert args.outlier_method == 'none'
    assert args.outlier_threshold == 1.5
    assert args.min_replications_for_filtering == 4
    assert args.replications is None
    assert not args.skip_plots


@pytest.mark.parametrize('stat', ['min', 'mean', 'median', 'max'])
def test_sector_groups_are_reduced_before_rank_statistics(tmp_path, stat):
    make_log(tmp_path)
    result = analysis(tmp_path, timing_stat=stat).summarize('simple')
    assert result.loc['Speciation', 2] == 9.  # max(8+1, 1+8), never max(8,1)+max(1,8)
    assert result.loc['TOTAL', 2] == 75.


@pytest.mark.parametrize('current', [False, True])
@pytest.mark.parametrize('style', ['auto', 'phase', 'grouped', 'simple', 'detailed', 'detail', 'raw'])
def test_historical_and_current_styles(tmp_path, current, style):
    make_log(tmp_path, current=current)
    result = analysis(tmp_path).summarize(style)
    assert result.shape[1] == 1
    assert not result.isna().any().any()
    if style == 'raw':
        assert 'Init/TOTAL' in result.index
        assert ('Write/write_time_step_emissions' in result.index) is current
    else:
        assert result.loc['TOTAL', 2] == 75.
        # All rank components are balanced in this example, so their statistics are additive too.
        assert result.drop(index='TOTAL')[2].sum() == pytest.approx(75.)
    if style in ('detail', 'detailed') and current:
        assert 'Writing/netcdf_write' in result.index
        assert 'Writing/write_time_step_emissions' not in result.index
        assert result.loc['Writing/Other', 2] == 1.


def test_zero_based_replications_and_whole_run_outliers(tmp_path):
    for rep, scale in enumerate([1., 1.01, 1.02, 1.03, 100.]):
        make_log(tmp_path, replication=rep, scale=scale)
    a = analysis(tmp_path, outlier_method='iqr')
    result = a.summarize('simple')
    assert result.loc['TOTAL', 2] == pytest.approx(75 * 1.015)
    assert a.outlier_summary['kept'].tolist() == [True, True, True, True, False]
    assert result.loc['Speciation', 2] == pytest.approx(9 * 1.015)
    filtered = analysis(tmp_path, replications=2, aggregation_method='median')
    assert len(filtered.logs) == 2
    assert filtered.summarize().loc['TOTAL', 2] == pytest.approx(75 * 1.005)


@pytest.mark.parametrize('method', ['iqr', 'mad', 'zscore'])
def test_filter_methods_and_minimum_samples(method):
    values = pd.Series([10., 11., 12., 13., 1000.])
    mask = ScalabilityAnalysis._get_inlier_mask(values, method, 1.5, 4)
    assert not mask.iloc[-1]
    assert ScalabilityAnalysis._get_inlier_mask(values, method, 1.5, 6).all()


def test_speedup_efficiency_and_zero_baseline():
    times = pd.DataFrame({1: [100., 0.], 2: [60., 0.], 4: [30., 0.]}, index=['TOTAL', 'communication'])
    speedup = ScalabilityAnalysis.get_speedup(times)
    efficiency = ScalabilityAnalysis.get_efficiency(times)
    assert speedup.loc['TOTAL', 4] == pytest.approx(100 / 30)
    assert efficiency.loc['TOTAL', 4] == pytest.approx(100 / 30 / 4)
    assert speedup.loc['communication'].isna().all()
    with pytest.raises(KeyError, match='Process count 1'):
        ScalabilityAnalysis.get_speedup(times.drop(columns=1))


def test_cli_skip_plots_writes_csvs(tmp_path):
    make_log(tmp_path, 1)
    make_log(tmp_path, 2, scale=.6)
    run(['--log-pattern', str(tmp_path / '<replication>/times_<processes>.csv'),
         '--out-dir', str(tmp_path / 'output'), '--summary-style', 'detailed', '--skip-plots'])
    assert {path.name for path in (tmp_path / 'output').iterdir()} == {
        'Times_Summary.csv', 'Times_Detail.csv', 'Outlier_Summary.csv',
        'Speedup_Summary.csv', 'Efficiency_Summary.csv'}
    frame = pd.read_csv(tmp_path / 'output/Speedup_Summary.csv', index_col=0)
    assert frame.loc['TOTAL', '2'] == pytest.approx(1 / .6)


def test_missing_serial_baseline_still_writes_times(tmp_path, capsys):
    make_log(tmp_path)
    paths = analysis(tmp_path).write_outputs(write_plots=False)
    assert len(paths) == 3
    assert 'Process count 1' in capsys.readouterr().err


def test_discovery_handles_named_repetitions_and_repeated_tokens(tmp_path):
    for name in ('first', 'second'):
        folder = tmp_path / name / '0002'
        folder.mkdir(parents=True)
        (folder / 'times_0002.csv').touch()
    records = ScalabilityAnalysis.discover_logs(str(tmp_path / '<replication>/<processes>/times_<processes>.csv'))
    assert len(records) == 2


def test_schema_and_bad_logs_are_rejected(tmp_path):
    path = make_log(tmp_path, current=False)
    with pytest.raises(ValueError, match='schema current'):
        analysis(tmp_path, timing_schema='current').summarize()
    frame = pd.read_csv(path)
    frame['0'] = -1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Invalid rank'):
        analysis(tmp_path).summarize()
    frame = frame.drop(columns='1')
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Expected ranks'):
        analysis(tmp_path).summarize()


def test_overlapping_timers_fail_instead_of_fabricating_residuals(tmp_path):
    path = make_log(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[frame.subsection == 'Speciation', ['0', '1']] = 1000
    frame.to_csv(path, index=False)
    a = analysis(tmp_path)
    with pytest.raises(ValueError, match='Overlapping'):
        a.summarize('simple')
    assert not a.summarize('raw').empty


def test_plot_outputs(tmp_path):
    make_log(tmp_path, 1)
    make_log(tmp_path, 2, scale=.6)
    outputs = analysis(tmp_path).write_outputs('simple', env='Test')
    images = [Path(path) for path in outputs if path.endswith('.png')]
    assert len(images) == 3
    assert all(path.stat().st_size > 1000 for path in images)


@pytest.mark.parametrize('kwargs', [{'replications': 0}, {'outlier_threshold': 0}, {'timing_schema': 'v3'}])
def test_invalid_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        analysis(tmp_path, **kwargs)
