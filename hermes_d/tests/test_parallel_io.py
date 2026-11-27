"""Configuration and output integration for selectable NetCDF parallel I/O."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from netCDF4 import Dataset
from nes import create_nes

from hermes_d.config import ConfigDelta


@pytest.fixture
def config_path(tmp_path):
    sectors = tmp_path / 'sectors.yaml'
    sectors.write_text('SECTORS: {}\n')
    config = tmp_path / 'hermes.ini'
    config.write_text('\n'.join([
        f'input_dir = {tmp_path}', f'data_path = {tmp_path}',
        f'output_dir = {tmp_path / "output"}', 'output_name = emissions.nc',
        'start_date = 2026/10/21 00:00:00', 'output_timestep_type = hourly',
        'output_timestep_num = 1', 'output_timestep_freq = 1',
        'domain_type = regular', 'inc_lat = 1', 'inc_lon = 1', 'n_lat = 2', 'n_lon = 2',
        f'auxiliary_files_path = {tmp_path / "aux"}', 'vertical_description = 100',
        f'sector_information = {sectors}', 'nuts2_shapefile = unused.shp', 'nuts2_info = unused.yaml',
    ]) + '\n')
    return config


def test_default_writing_configuration(config_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['hermes_delta'])
    options = ConfigDelta(config_path=str(config_path)).options
    assert options.serial_write is True
    assert options.parallel_io_mode == 'independent'


@pytest.mark.parametrize('mode', ['independent', 'collective'])
def test_ini_and_cli_override(config_path, monkeypatch, mode):
    with config_path.open('a') as config:
        config.write('serial_write = False\nparallel_io_mode = collective\n')
    monkeypatch.setattr('sys.argv', ['hermes_delta', '--parallel_io_mode', mode])
    options = ConfigDelta(config_path=str(config_path)).options
    assert options.serial_write is False
    assert options.parallel_io_mode == mode


def test_invalid_parallel_mode(config_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['hermes_delta', '--parallel_io_mode', 'invalid'])
    with pytest.raises(SystemExit) as error:
        ConfigDelta(config_path=str(config_path))
    assert error.value.code == 2


@pytest.mark.parametrize('timing_enabled', [False, True])
@pytest.mark.parametrize('serial', [True, False])
@pytest.mark.parametrize('mode', ['independent', 'collective'])
def test_main_passes_writing_options(monkeypatch, tmp_path, serial, mode, timing_enabled):
    import hermes_d.hermes as module
    monkeypatch.setattr(module.hermes_d, "DEBUG", False)
    monkeypatch.setattr(module.hermes_d, "WRITE_TIME_LOG", timing_enabled)
    app = module.HermesDelta.__new__(module.HermesDelta)
    app.first_time = False
    app.start_date = datetime(2026, 10, 21)
    app.size = 1
    app.time_step_list = [app.start_date, app.start_date]
    app.out_format = 'DEFAULT'
    app.options = SimpleNamespace(serial_write=serial, parallel_io_mode=mode,
                                  output_dir=str(tmp_path), end_date=app.start_date)
    app.config = Mock()
    app.config.get_output_name.return_value = str(tmp_path / 'emissions.nc')
    result = Mock()
    diagnostics = dict(unit_conversion=1., communication=2., netcdf_write=3., netcdf_write_unaccounted=4.)
    result.append_time_step_data.return_value = diagnostics if timing_enabled else None
    app.prepare_output = Mock(return_value=result)
    app.prepare_inventories = Mock()
    app.get_time_step_emissions = Mock(return_value={})
    monkeypatch.setattr(module, 'full_time', 0, raising=False)
    for name in ['log_message', 'record_time', 'finalize_time_log']:
        monkeypatch.setattr(module, name, Mock())
    assert app.main() is None
    result.to_netcdf.assert_called_once_with(
        str(tmp_path / 'emissions.nc'), serial=serial,
        keep_open=True, nc_type='DEFAULT', parallel_io_mode=mode)
    expected = {'out_format': 'DEFAULT'}
    if timing_enabled:
        expected['return_timing'] = True
    from unittest.mock import call
    assert result.append_time_step_data.call_args_list == [call(0, **expected), call(1, **expected)]
    records = module.record_time.call_args_list
    for name, value in diagnostics.items():
        entries = [args.args[2] for args in records if args.args[:2] == ('Write', name)]
        assert entries == ([value, value] if timing_enabled else [])
    times = {}
    for args in records:
        section, name, value = args.args
        if section == 'Write':
            times[name] = times.get(name, 0) + value
    assert times['TOTAL'] == pytest.approx(sum(times[name] for name in (
        'prepare_output', 'write_time_step_emissions', 'close_output')))
    result.close.assert_called_once()


@pytest.mark.parametrize('mode,unlimited', [('independent', False), ('collective', True)])
def test_nes_time_step_output(tmp_path, mode, unlimited):
    """Exercise the NES writer on one rank, including time dimension semantics."""
    times = [datetime(2026, 10, 21), datetime(2026, 10, 21, 1)]
    grid = create_nes(projection='regular', lat_orig=40, lon_orig=0,
                      inc_lat=1, inc_lon=1, n_lat=2, n_lon=2, times=times)
    grid.set_levels({'data': np.array([100.]), 'units': 'm', 'positive': 'up'})
    grid.variables = {'emissions': {'data': None, 'dtype': np.float64, 'units': 'kg'}}
    output = str(tmp_path / f'{mode}.nc')
    grid.to_netcdf(output, serial=False, parallel_io_mode=mode, keep_open=True)
    try:
        for step in range(2):
            grid.variables['emissions']['data'] = np.full((1, 1, 2, 2), step + 1.)
            timing = grid.append_time_step_data(step, return_timing=True)
            assert set(timing) == {'unit_conversion', 'communication', 'netcdf_write',
                                   'netcdf_write_unaccounted', 'total'}
            assert all(value >= 0 for value in timing.values())
            assert timing['total'] == pytest.approx(sum(timing[name] for name in timing if name != 'total'))
    finally:
        grid.close()
    with Dataset(output) as dataset:
        assert dataset.dimensions['time'].isunlimited() is unlimited
        np.testing.assert_array_equal(dataset.variables['emissions'][:, 0],
                                      np.array([np.ones((2, 2)), np.full((2, 2), 2.)]))


def test_sector_combination_preserves_emissions_and_records_separate_timers(monkeypatch):
    import hermes_d.hermes as module
    record = Mock()
    monkeypatch.setattr(module, 'record_time', record)
    app = module.HermesDelta.__new__(module.HermesDelta)
    first = {'NOx': {'data': np.ones((1, 1, 2, 2)), 'units': 'kg'}}
    second = {'NOx': {'data': np.full((1, 1, 2, 2), 2.), 'units': 'kg'}}
    app.sector_list = [SimpleNamespace(name='SNAP01', get_4d_emissions=Mock(return_value=first)),
                       SimpleNamespace(name='SNAP02', get_4d_emissions=Mock(return_value=second))]
    output = app.get_time_step_emissions(datetime(2026, 10, 21), 0)
    np.testing.assert_array_equal(output['NOx']['data'], np.full((1, 1, 2, 2), 3.))
    labels = [entry.args[:2] for entry in record.call_args_list]
    assert labels.count(('Calcul', 'combine_sectors')) == 2
    assert ('Calcul', 'get_time_step_emissions') in labels
    assert ('SNAP01', 'get_4d_emissions') in labels
    assert ('SNAP02', 'get_4d_emissions') in labels
