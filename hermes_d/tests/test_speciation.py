"""Self-contained MONARCH speciation tests with small real SNAP01 examples.

Source: HERMES_data/hermesv3_D, SNAP01 profile E_010104 and annual inventories
2021/edition2023; individual_species_CB05_AERO5sindhant_MONARCH.csv (not AERO6).
Only relevant mechanism columns are retained below. No external files are needed.
Expected arithmetic is explicit: weight/100 / MWt * coefficient * 1000.
These are regression checks, not scientific validation of the NO2 mass basis or
of PM2.5 totals (the source includes a total plus components, summing to 187.2%).
"""
from io import StringIO

import pandas as pd
import pytest

from hermes_d.modules.speciation import SpeciationDelta, speciate

# Actual component rows from E_010104; weights and coefficients are unmodified.
PROFILE = """P_spec,CAS,Species_name,Weight_percentual,MWt,Specie_ori
E_010104,10102-43-9,nitrogen monoxide,90,30.01,nox_no2
E_010104,10102-44-0,nitrogen dioxide,10,46.01,nox_no2
E_010104,50-00-0,formaldehyde,100,30.03,nmvoc
E_010104,630-08-0,carbon monoxide,100,28.01,co
E_010104,pm25,pm25,100,1,pm25
E_010104,pm25_oc,pm25_oc,20.1,1,pm25
E_010104,pm25_other,pm25_other,66.4,1,pm25
E_010104,pm25_so4,pm25_so4,0.7,1,pm25
"""
MECHANISM = """CAS,Species_name,NO,NO2,FORM,CO,SOAP_anthro,POA,PMFINE,PSO4,PMC
units,,mol,mol,mol,mol,kg,kg,kg,kg,kg
short_description,,NO,NO2,formaldehyde,CO,SOAP,organic,other,sulphate,coarse
10102-43-9,nitrogen monoxide,1,0,0,0,0,0,0,0,0
10102-44-0,nitrogen dioxide,0,1,0,0,0,0,0,0,0
50-00-0,formaldehyde,0,0,1,0,0,0,0,0,0
630-08-0,carbon monoxide,0,0,0,1,1,0,0,0,0
pm25,pm25,0,0,0,0,0,0,0,0,0
pm25_oc,pm25_oc,0,0,0,0,0,1.3,0,0,0
pm25_other,pm25_other,0,0,0,0,0,0,1,0,0
pm25_so4,pm25_so4,0,0,0,0,0,0,0,1,0
"""
POLLUTANTS = ['nox_no2', 'nmvoc', 'pm25', 'co']


def read_csv(text):
    """Read inline tables, keeping the examples visible next to the assertions."""
    return pd.read_csv(StringIO(text))


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Write temporary inputs and capture warnings; create no permanent test files."""
    profiles = tmp_path / 'profiles.csv'
    mechanism = tmp_path / 'mechanism.csv'
    profiles.write_text(PROFILE)
    mechanism.write_text(MECHANISM)
    warnings = []
    monkeypatch.setattr('hermes_d.DEBUG', False)
    monkeypatch.setattr('hermes_d.modules.speciation.speciation.log_message',
                        lambda message, level: warnings.append(message) if level == 7 else None)
    monkeypatch.setattr('hermes_d.modules.speciation.report.log_message', lambda *args, **kwargs: None)
    return profiles, mechanism, warnings


@pytest.mark.parametrize('kind,amounts', [
    ('sector', [560.61, 41.61, 16.64, 145.9]),  # 010104 / ES11
    ('point-source', [157.31, 12.41708104, 0.23, 1.68]),  # LPS-00159 / ES51
])
@pytest.mark.parametrize('legacy_ratio', [None, 1.0, 2.0, float('nan')])
def test_monarch_conversion(case, kind, amounts, legacy_ratio):
    """Real NOx split, formaldehyde, CO and aerosols; obsolete ratios cannot change results.

    Input amounts are t/year. Gas factors yield kmol/year before output processing;
    particle factors yield kg/year. CO also feeds SOAP_anthro in the source mapping.
    PMC has no contribution and stays zero. Original columns disappear; metadata stays.
    """
    profile_path, mechanism_path, warnings = case
    profiles = read_csv(PROFILE)
    if legacy_ratio is not None:
        profiles['TOG_to_VOC_ratio'] = legacy_ratio
    profiles.to_csv(profile_path, index=False)
    records = pd.DataFrame([amounts], columns=POLLUTANTS).assign(P_spec='E_010104', inventory=kind)
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None,
                           records if kind == 'sector' else records.iloc[:0].copy(),
                           records if kind == 'point-source' else None)
    sector, points, species, factors = proc.initialize(POLLUTANTS, 'SNAP01')
    result = speciate(sector if points is None else points, factors, species, POLLUTANTS)
    nox, voc, pm, co = amounts
    expected = [nox * 900 / 30.01, nox * 100 / 46.01, voc * 1000 / 30.03,
                co * 1000 / 28.01, co * 1000 / 28.01,
                pm * 201 * 1.3, pm * 664, pm * 7, 0.0]
    assert result[species].iloc[0].tolist() == pytest.approx(expected, rel=1e-12)
    assert result.columns.tolist() == ['inventory'] + species
    assert result.inventory.tolist() == [kind]
    assert len(warnings) == 1 and "components don't sum up to 100" in warnings[0]


def test_equal_profiles_share_id(case):
    """Copy a real profile under another ID; check exact deduplication and row-order remapping."""
    profile_path, mechanism_path, _ = case
    profiles = read_csv(PROFILE)
    pd.concat([profiles, profiles.assign(P_spec='COPY')]).to_csv(profile_path, index=False)
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, None)
    factors, ids, _ = proc.get_speciation_mapping(POLLUTANTS)
    assert len(factors) == 1 and ids['E_010104'] == ids['COPY']
    records = pd.DataFrame({'P_spec': ['COPY', 'E_010104']})
    assert proc.map_emission_p_spec(records, ids, 'SNAP01').P_spec.tolist() == [ids['COPY']] * 2


@pytest.mark.parametrize('column,label', [(None, 'data (unrecognized file)'),
                                          ('is_point_source', 'SEIE'), ('Code', 'LPS')])
@pytest.mark.parametrize('profile_id', ['unknown', None, 42])
def test_unknown_profile_is_error(column, label, profile_id):
    """Unknown/null/numeric IDs raise descriptive errors for both inventory types without mutation."""
    records = pd.DataFrame({'P_spec': [profile_id]})
    if column is not None:
        records[column] = False
    original = records.copy(deep=True)
    with pytest.raises(ValueError) as error:
        SpeciationDelta(None, None, None, None).map_emission_p_spec(records, {}, 'SNAP01')
    assert str(error.value) == f'Found P_spec {profile_id} in {label} but not in speciation_SNAP01.csv.'
    pd.testing.assert_frame_equal(records, original)


def test_missing_co_reports_amount(case):
    """Inject missing CO coverage, retaining a complete copy to defeat the global presence check.

    Real E_010104 CO amounts: regional 145.9+83.059=228.959; facilities
    1.68+102.9=104.58 t/year. Each warning counts two records. A covered copy
    still converts normally; only uncovered CO/SOAP_anthro become zero.
    """
    profile_path, mechanism_path, warnings = case
    profiles = read_csv(PROFILE)
    pd.concat([profiles.loc[profiles.Specie_ori != 'co'],
               profiles.assign(P_spec='COMPLETE')]).to_csv(profile_path, index=False)
    sector = pd.DataFrame({'P_spec': ['E_010104'] * 2 + ['COMPLETE'], 'co': [145.9, 83.059, 1.0]})
    points = pd.DataFrame({'P_spec': ['E_010104'] * 2, 'co': [1.68, 102.9]})
    for records in [sector, points]:
        records[['nox_no2', 'nmvoc', 'pm25']] = 0.0
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, sector, points)
    sector, points, species, factors = proc.initialize(POLLUTANTS, 'SNAP01')
    prefix = "Speciation profile 'E_010104' in 'profiles.csv' does not include original pollutant 'co'. "
    suffix = ' records will not contribute to the speciated output.'
    assert [message for message in warnings if 'does not include' in message] == [
        prefix + '228.959 t/year from 2 sector inventory' + suffix,
        prefix + '104.58 t/year from 2 point-source inventory' + suffix]
    result = speciate(sector, factors, species, POLLUTANTS).sort_index()
    assert result.CO.tolist() == pytest.approx([0, 0, 1000 / 28.01])
    assert result.SOAP_anthro.tolist() == pytest.approx(result.CO.tolist())
    assert speciate(points, factors, species, POLLUTANTS)[species].eq(0).all().all()


def test_warning_scope(case):
    """Zero emissions and unused profiles stay quiet; point-only profiles are checked."""
    profile_path, mechanism_path, warnings = case
    profiles = pd.DataFrame({'P_spec': ['A', 'B', 'UNUSED'], 'CAS': ['c1'] * 3,
                             'Specie_ori': ['co'] * 3}).set_index(['P_spec', 'CAS'])
    sector = pd.DataFrame({'P_spec': ['A', 'unknown'], 'co': [0.0, 1.0], 'nh3': [0.0, 8.0]})
    points = pd.DataFrame({'P_spec': ['B'], 'co': [1.0], 'nh3': [2.5]})
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, sector, points)
    proc._warn_missing_profile_pollutants(profiles, ['co', 'nh3'])
    assert warnings == [
        "Speciation profile 'B' in 'profiles.csv' does not include original pollutant 'nh3'. "
        "2.5 t/year from 1 point-source inventory records will not contribute to the speciated output."]


def test_unselected_pollutants_are_warned_and_excluded(case):
    """Selecting only CO must ignore other components, including invalid MWt and unknown CAS.

    An unselected component deliberately maps to CO as well, ensuring exclusion
    precedes the multiple-origin check. A profile containing only unselected rows
    keeps its original ID, and selected CO emissions there generate a coverage warning.
    """
    profile_path, mechanism_path, warnings = case
    profiles = read_csv(PROFILE)
    extra = profiles.loc[profiles.Specie_ori == 'co'].assign(P_spec='UNSELECTED', Specie_ori='nh3')
    profiles = pd.concat([profiles, extra], ignore_index=True)
    profiles.loc[profiles.Specie_ori == 'nmvoc', ['CAS', 'MWt']] = ['unknown-cas', float('nan')]
    profiles.to_csv(profile_path, index=False)
    records = pd.DataFrame({'P_spec': ['E_010104', 'UNSELECTED'], 'co': [145.9, 3.0]})
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, records)
    sector, _, species, factors = proc.initialize(['co'], 'SNAP01')
    assert warnings == [
        'Pollutants nh3, nmvoc, nox_no2, pm25 in profiles.csv are not selected in the sector configuration. '
        'Their profile rows will be excluded from validation and speciation factors.',
        "Speciation profile 'UNSELECTED' in 'profiles.csv' does not include original pollutant 'co'. "
        '3 t/year from 1 sector inventory records will not contribute to the speciated output.',
    ]
    assert all(set(origins) == {'co'} for profile in factors.values() for origins in profile.values())
    result = speciate(sector, factors, species, ['co']).sort_index()
    assert result.CO.tolist() == pytest.approx([145.9 * 1000 / 28.01, 0.0])
    assert result.SOAP_anthro.tolist() == pytest.approx(result.CO.tolist())
    assert result.drop(columns=['CO', 'SOAP_anthro']).eq(0).all().all()


def test_selected_inconsistent_molecular_weight_remains_error(case):
    """Relaxing missing MWt must not allow conflicting known molecular weights."""
    profile_path, mechanism_path, _ = case
    profiles = read_csv(PROFILE)
    conflict = profiles.loc[profiles.Specie_ori == 'co'].assign(P_spec='OTHER', MWt=99.0)
    pd.concat([profiles, conflict]).to_csv(profile_path, index=False)
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, None)
    with pytest.raises(ValueError, match="don't have unique MWt"):
        proc.get_speciation_mapping(['co'])


@pytest.mark.parametrize('missing_fields', [('CAS',), ('MWt',), ('CAS', 'MWt')])
@pytest.mark.parametrize('resolved_weight', [70.0, 0.0])
def test_unresolved_components_count_in_totals_but_not_conversion(case, missing_fields, resolved_weight):
    """A real CO component plus unresolved remainder sums to 100%, without normalization.

    Exercise missing CAS, MWt and both, including an entirely unresolved profile.
    Multiple anonymous rows are allowed. Only the resolved fraction reaches both
    CO-derived MONARCH species; no false weight-total or missing-profile error occurs.
    """
    profile_path, mechanism_path, warnings = case
    component = read_csv(PROFILE).query("Specie_ori == 'co'")
    resolved = component.assign(Weight_percentual=resolved_weight)
    unknown = component.assign(Weight_percentual=(100 - resolved_weight) / 2,
                               Species_name='unresolved component')
    for field in missing_fields:
        unknown[field] = float('nan') if field == 'MWt' else 'unspecified'
    if 'CAS' not in missing_fields:
        unknown['CAS'] = 'S1-99121'
    # Distinct CAS are needed for two known-ID components; anonymous rows need no fake IDs.
    unknown2 = unknown.copy()
    if 'CAS' not in missing_fields:
        unknown2['CAS'] = 'S1-99122'
    rows = [unknown, unknown2] if resolved_weight == 0 else [resolved, unknown, unknown2]
    pd.concat(rows).to_csv(profile_path, index=False)
    records = pd.DataFrame({'P_spec': ['E_010104'], 'co': [145.9]})
    from hermes_d.modules.speciation.report import SpeciationReport
    output = profile_path.parent / 'issues.csv'
    report = SpeciationReport(output)
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, records, report=report)
    sector, _, species, factors = proc.initialize(['co'], 'SNAP01')
    diagnostic_rows = pd.read_csv(output)
    assert len(diagnostic_rows) == 2
    assert diagnostic_rows.total_weight_percent.tolist() == [100.0, 100.0]
    assert diagnostic_rows.excluded_weight_percent.tolist() == [(100 - resolved_weight) / 2] * 2
    if 'CAS' in missing_fields:
        assert diagnostic_rows.component_id.isna().all()
    assert len(warnings) == 2
    for message in warnings:
        assert f"has missing {' and '.join(missing_fields)}" in message
        assert 'retained in composition totals but excluded from conversion factors' in message
        assert f'weight ({(100 - resolved_weight) / 2:g}%)' in message
    result = speciate(sector, factors, species, ['co'])
    expected = 145.9 * resolved_weight / 100 * 1000 / 28.01
    assert result.CO.iloc[0] == pytest.approx(expected)
    assert result.SOAP_anthro.iloc[0] == pytest.approx(expected)
    assert result.drop(columns=['CO', 'SOAP_anthro']).eq(0).all().all()


@pytest.mark.parametrize('fault,issue_type', [
    ('weight', 'weight_total_deviation'), ('unresolved', 'missing_component_data'),
    ('unselected', 'unselected_pollutant'), ('absent', 'missing_file_pollutant'),
    ('coverage', 'missing_profile_pollutant'), ('duplicate', 'duplicate_component'),
    ('molecular_weight', 'inconsistent_molecular_weight'), ('cas', 'missing_mechanism_cas'),
    ('inventory', 'missing_inventory_profile'), ('origins', 'multiple_original_pollutants'),
])
def test_report_contains_validation_context(case, tmp_path, monkeypatch, fault, issue_type):
    """Exercise every diagnostic type; CSV survives errors outside DEBUG with original context."""
    from hermes_d.modules.speciation.report import SpeciationReport
    monkeypatch.setattr('hermes_d.modules.speciation.report.log_message', lambda *args, **kwargs: None)
    profile_path, mechanism_path, _ = case
    profiles = read_csv(PROFILE).query("Specie_ori == 'co'").copy()
    records = pd.DataFrame({'P_spec': ['E_010104'], 'co': [145.9]})
    selected = ['co']
    fatal = fault in {'duplicate', 'molecular_weight', 'cas', 'inventory', 'origins'}
    if fault == 'weight':
        profiles['Weight_percentual'] = 80.0
    elif fault == 'unresolved':
        profiles['MWt'] = float('nan')
    elif fault == 'unselected':
        profiles = pd.concat([profiles, profiles.assign(Specie_ori='nh3', P_spec='EXTRA')])
    elif fault in {'absent', 'coverage'}:
        records['nh3'] = 4.5
        selected.append('nh3')
        if fault == 'coverage':
            profiles = pd.concat([profiles, profiles.assign(Specie_ori='nh3', P_spec='UNUSED')])
    elif fault == 'duplicate':
        profiles = pd.concat([profiles, profiles])
    elif fault == 'molecular_weight':
        profiles = pd.concat([profiles, profiles.assign(P_spec='OTHER', MWt=99.0)])
    elif fault == 'cas':
        profiles['CAS'] = 'unknown-cas'
    elif fault == 'inventory':
        records['P_spec'] = 'unknown-profile'
    elif fault == 'origins':
        profiles = pd.concat([profiles, profiles.assign(CAS='10102-43-9', Specie_ori='nox_no2', MWt=30.01)])
        mechanism = read_csv(MECHANISM)
        mechanism.loc[mechanism.CAS == '10102-43-9', 'CO'] = '1'
        mechanism.to_csv(mechanism_path, index=False)
        selected.append('nox_no2')
    profiles.to_csv(profile_path, index=False)
    output = tmp_path / 'issues.csv'
    report = SpeciationReport(output)
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None, records, report=report,
                           inventory_files={'sector inventory': 'original_sector.csv'})
    if fatal:
        with pytest.raises(ValueError):
            proc.initialize(selected, 'SNAP01')
    else:
        proc.initialize(selected, 'SNAP01')
    rows = pd.read_csv(output).fillna('')
    relevant = rows.loc[rows.issue_type == issue_type]
    assert not relevant.empty
    assert set(relevant.sector) == {'SNAP01'}
    assert set(relevant.source_file) == {str(profile_path)}
    assert set(relevant.severity) == {'error' if fatal else 'warning'}
    assert relevant.message.ne('').all() and relevant.impact.ne('').all()
    if fault == 'weight':
        assert relevant.total_weight_percent.tolist() == [80.0]
    if fault == 'unresolved':
        assert relevant.excluded_weight_percent.tolist() == [100.0]
        assert relevant.component_id.tolist() == ['630-08-0']
    if fault in {'coverage', 'inventory'}:
        assert relevant.inventory_file.tolist() == ['original_sector.csv']
    if fault == 'coverage':
        assert relevant.emission_amount_t_year.tolist() == [4.5]
        assert relevant.affected_records.tolist() == [1]


def test_report_master_only_reset_and_multiple_sectors(case, tmp_path, monkeypatch):
    """One report spans sectors; worker processes never write; clean reruns remove stale CSVs."""
    from hermes_d.modules.speciation.report import SpeciationReport
    summaries = []
    monkeypatch.setattr('hermes_d.modules.speciation.report.log_message',
                        lambda message, level: summaries.append(message))
    profile_path, mechanism_path, _ = case
    output = tmp_path / 'shared.csv'
    report = SpeciationReport(output)
    for sector_name in ['SNAP01', 'SNAP02']:
        proc = SpeciationDelta(str(profile_path), str(mechanism_path), None,
                               pd.DataFrame(columns=['P_spec'] + POLLUTANTS), report=report)
        proc.initialize(POLLUTANTS, sector_name)
    assert pd.read_csv(output).sector.tolist() == ['SNAP01', 'SNAP02']
    assert '2 warning issues, 0 error issues' in summaries[-1]
    assert str(output) in summaries[-1]
    original = output.read_bytes()
    worker = SpeciationReport(output, master=False)
    worker.add(severity='error', message='worker must not write')
    worker.summary()
    assert output.read_bytes() == original
    clean = SpeciationReport(output)
    # Use a complete single-contaminant profile to avoid all diagnostics.
    read_csv(PROFILE).query("Specie_ori == 'co'").to_csv(profile_path, index=False)
    proc = SpeciationDelta(str(profile_path), str(mechanism_path), None,
                           pd.DataFrame({'P_spec': ['E_010104'], 'co': [1.0]}), report=clean)
    proc.initialize(['co'], 'SNAP01')
    assert not output.exists()


@pytest.mark.parametrize('output_name', ['emissions.nc', 'emissions.v1_<YYYYMMDD>.nc'])
def test_report_is_configured_in_output_directory(tmp_path, monkeypatch, output_name):
    """Derive the report from the resolved NetCDF output name, preserving directory and date tokens."""
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module('hermes_d.hermes')
    paths = []
    monkeypatch.setattr(module, 'SpeciationReport', lambda output_path, master: paths.append(output_path))
    monkeypatch.setattr(module, 'log_message', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'record_time', lambda *args, **kwargs: None)
    application = module.HermesDelta.__new__(module.HermesDelta)
    application.master = True
    from datetime import datetime
    from hermes_d.config import ConfigDelta
    application.start_date = datetime(2026, 10, 20)
    application.options = SimpleNamespace(individual_speciation_file='mechanism.csv',
                                          output_dir=str(tmp_path / 'output'),
                                          auxiliary_files_path=str(tmp_path / 'auxiliary'),
                                          sector_information={'SECTORS': {}},
                                          output_name=output_name, output_timestep_type='daily')
    application.config = ConfigDelta.__new__(ConfigDelta)
    application.config.options = application.options
    assert application.make_sector_list() == []
    expected_name = output_name.replace('<YYYYMMDD>', '20261020')[:-3] + '_speciation_issues.csv'
    assert paths == [str(tmp_path / 'output' / expected_name)]
