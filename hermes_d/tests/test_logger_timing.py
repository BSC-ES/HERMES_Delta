"""Timing activation, MPI summaries and configurable output paths."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import hermes_d
from hermes_d.config import ConfigDelta, logger_config as logger
from hermes_d.tests.test_parallel_io import config_path  # noqa: F401


@pytest.fixture(autouse=True)
def reset_timing(monkeypatch):
    monkeypatch.setattr(hermes_d, 'DEBUG', False)
    monkeypatch.setattr(hermes_d, 'WRITE_TIME_LOG', False)
    for name in ('time_records', 'memory_records'):
        monkeypatch.setattr(logger, name, pd.Series(
            dtype=float, index=pd.MultiIndex.from_tuples([], names=['section', 'subsection']), name=0))
    monkeypatch.setattr(logger, 'log_message', Mock())


@pytest.mark.parametrize('level,enabled,expected', [(3, False, False), (3, True, True), (5, False, True)])
def test_configuration_activation(config_path, monkeypatch, level, enabled, expected):
    monkeypatch.setattr('sys.argv', ['hermes_delta', '--log_level', str(level), '--write_time_log', str(enabled)])
    config = ConfigDelta(config_path=str(config_path))
    assert config.options.time_log_path is None
    assert hermes_d.DEBUG is (level == 5)
    assert hermes_d.WRITE_TIME_LOG is expected


def test_ini_options_and_cli_override(config_path, monkeypatch):
    with config_path.open('a') as config:
        config.write('write_time_log = True\ntime_log_path = ini.csv\n')
    monkeypatch.setattr('sys.argv', ['hermes_delta', '--time_log_path', 'cli_<date>_<processes>.csv'])
    config = ConfigDelta(config_path=str(config_path))
    assert hermes_d.WRITE_TIME_LOG is True
    assert config.get_time_log_path(datetime(2026, 10, 21), 32).endswith('cli_2026102100_0032.csv')


@pytest.mark.parametrize('debug,enabled', [(False, False), (False, True), (True, False)])
def test_recording_and_master_csv(tmp_path, monkeypatch, debug, enabled):
    monkeypatch.setattr(hermes_d, 'DEBUG', debug)
    monkeypatch.setattr(hermes_d, 'WRITE_TIME_LOG', enabled)
    comm = Mock()
    monkeypatch.setattr(logger, 'my_comm', comm)
    monkeypatch.setattr(logger, 'my_rank', 0)
    logger.record_time('Init', 'TOTAL', 2.)
    logger.record_time('Init', 'TOTAL', 3.)
    logger.record_memory('Init', 'TOTAL', 7.)
    assert len(logger.memory_records) == int(debug)
    timing = debug or enabled
    assert len(logger.time_records) == int(timing)
    if timing:
        other = logger.time_records.copy().rename(1) * 2
        comm.gather.return_value = [logger.time_records, other]
    output = tmp_path / 'nested' / 'times.csv'
    logger.finalize_time_log(str(output))
    assert output.exists() is timing
    if timing:
        summary = pd.read_csv(output, index_col=[0, 1])
        assert summary.loc[('Init', 'TOTAL'), ['min', 'mean', 'max']].tolist() == [5., 7.5, 10.]
    else:
        comm.gather.assert_not_called()
        assert not output.parent.exists()


def test_non_master_only_participates_in_gather(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_d, 'WRITE_TIME_LOG', True)
    comm = Mock()
    monkeypatch.setattr(logger, 'my_comm', comm)
    monkeypatch.setattr(logger, 'my_rank', 1)
    output = tmp_path / 'nested' / 'times.csv'
    logger.finalize_time_log(str(output))
    comm.gather.assert_called_once()
    assert not output.parent.exists()


def test_time_logging_does_not_enable_memory_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_d, 'WRITE_TIME_LOG', True)
    comm = Mock()
    monkeypatch.setattr(logger, 'my_comm', comm)
    logger.finalize_mem_log(str(tmp_path / 'memory.csv'))
    comm.gather.assert_not_called()
    assert not (tmp_path / 'memory.csv').exists()


@pytest.mark.parametrize('frequency,stamp', [
    ('hourly', '2026102100'), ('daily', '20261021'), ('monthly', '202610'), ('yearly', '2026')])
def test_paths_preserve_default_and_resolve_templates(tmp_path, frequency, stamp):
    config = ConfigDelta.__new__(ConfigDelta)
    config.options = SimpleNamespace(output_dir=str(tmp_path), output_name='emissions_<date>.nc',
                                     output_timestep_type=frequency, time_log_path=None)
    date = datetime(2026, 10, 21)
    assert config.get_time_log_path(date, 32) == str(tmp_path / 'logs' / f'emissions_{stamp}_Times_0032.csv')
    config.options.time_log_path = 'timing/<YYYY>/<date>_<processes>.csv'
    assert config.get_time_log_path(date, 8) == str(tmp_path / 'timing' / '2026' / f'{stamp}_0008.csv')
    config.options.time_log_path = str(tmp_path / 'absolute' / 'times_<YYYYMMDDHH>.csv')
    assert config.get_time_log_path(date, 8) == str(tmp_path / 'absolute' / 'times_2026102100.csv')
