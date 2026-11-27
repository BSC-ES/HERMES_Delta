"""Build profile-specific chemical conversion factors and apply them to inventories.

Sector initialization prepares factors and remaps P_spec identifiers. Actual
conversion runs before vertical, temporal and horizontal emission distribution.
"""

from typing import Dict, Tuple, List, Optional
from pandas import DataFrame, read_csv, concat, isna
from os import path, makedirs
import hermes_d
from hermes_d.config import precision, log_message
from .report import SpeciationReport


class SpeciationDelta(object):
    def __init__(self, spec_snap_path: str, indi_spec_path: str, spec_aux_path: str,
                 sector_info: DataFrame,
                 point_sources: Optional[DataFrame] = None, report: Optional[SpeciationReport] = None,
                 master: bool = True, inventory_files: Optional[Dict] = None):
        """
        Store input paths and emission tables for later initialization.

        Parameters
        ----------
        spec_snap_path : str
            Sector profile CSV with P_spec, CAS, Species_name, Specie_ori,
            Weight_percentual and MWt columns.
        indi_spec_path : str
            Mechanism CSV: CAS and Species_name followed by species columns.
            The first two data rows contain units and descriptions; subsequent
            rows contain numeric conversion coefficients indexed by CAS.
        spec_aux_path : str
            Diagnostic mapping output path, written only when hermes_d.DEBUG is enabled.
        sector_info : DataFrame
            Sector inventory with P_spec and original pollutant columns.
        point_sources : DataFrame, optional
            Point-source inventory with the same speciation fields, or None.

        report : SpeciationReport, optional
            Run-wide issue collector, configured by HermesDelta in output_dir.
            Reports are independent of DEBUG. Without a collector, no CSV is written.
        master : bool, optional
            Whether this process owns report output when no shared collector is supplied.
        inventory_files : dict, optional
            Original paths keyed by 'sector inventory' and 'point-source inventory'.

        Notes
        -----
        No profile files are read and no factors are calculated until initialize() is called.
        """
        self.spec_snap_path = spec_snap_path
        self.indi_spec_path = indi_spec_path
        self.spec_aux_path = spec_aux_path
        self.report = report
        self.master = master
        self.inventory_files = inventory_files or {}
        self.sector_name = ''
        self._profile_totals = {}

        self.speciation_mapping = None
        self.species_list = None

        self.sector_info = sector_info
        self.point_sources = point_sources

    def initialize(self, pollutants, sector_name) -> Tuple[DataFrame, Optional[DataFrame], List, Dict]:
        """
        Build conversion factors and update P_spec in the stored sector_info and
        point_sources DataFrames with new speciation profile IDs.
        Parameters
        ----------
        pollutants: list
            A list of pollutants provided in Sector config file.
        sector_name: str
            The name of the sector being processed.

        Returns
        -------
        Tuple[DataFrame, DataFrame or None, List, Dict]
            Updated sector_info DataFrame, updated point_sources DataFrame (if provided), list of species
            in the chemical mechanism, and the speciation mapping dictionary.
        """
        self.sector_name = sector_name
        try:
            self.speciation_mapping, mapping_p_spec, self.species_list = self.get_speciation_mapping(pollutants)

            if hermes_d.DEBUG and self.master:
                log_message("Saving speciation profile in {0}".format(self.spec_aux_path), level=2)
                makedirs(path.dirname(self.spec_aux_path), exist_ok=True)
                with open(self.spec_aux_path, "w") as file:
                    for spec_id, value in self.speciation_mapping.items():
                        file.write("{0}: \n".format(spec_id))
                        for spec, orig in value.items():
                            file.write("    {} :  {}\n".format(spec, orig))

            self.sector_info = self.map_emission_p_spec(self.sector_info, mapping_p_spec, sector_name)
            if self.point_sources is not None:
                self.point_sources = self.map_emission_p_spec(self.point_sources, mapping_p_spec, sector_name)

            return self.sector_info, self.point_sources, self.species_list, self.speciation_mapping
        finally:
            if self.report is not None:
                self.report.summary()

    def _record_issue(self, severity, issue_type, message, action, impact, **context):
        """Record structured context without changing logging or validation severity."""
        if self.report is not None:
            inventory_type = context.get('inventory_type')
            self.report.add(inventory_file=self.inventory_files.get(inventory_type),
                            severity=severity, issue_type=issue_type, sector=self.sector_name,
                            source_file=self.spec_snap_path, message=message, action=action,
                            impact=impact, **context)

    def _record_components(self, rows, severity, issue_type, message, action, impact, **context):
        """Expand component diagnostics into CSV rows with original profile IDs."""
        for _, row in rows.iterrows():
            details = dict(context)
            if action == 'exclude_unselected':
                details['excluded_weight_percent'] = row['Weight_percentual']
            self._record_issue(
                severity, issue_type, message, action, impact,
                profile_id=row['P_spec'], original_pollutant=row['Specie_ori'],
                component_id=row['CAS'], component_name=row['Species_name'],
                total_weight_percent=self._profile_totals.get((row['P_spec'], row['Specie_ori'])), **details)

    # noinspection DuplicatedCode
    def get_speciation_mapping(self, pollutants) -> Tuple[Dict, Dict, List]:
        """
        Build and deduplicate conversion factors from the two input CSVs.

        Parameters
        ----------
        pollutants : list of str
            Original pollutants selected in the sector configuration.

        Returns
        -------
        factors : dict
            Nested mapping {new_profile_id: {species: {original_pollutant: factor}}}.
            Each nonempty species mapping contains exactly one original pollutant.
        profile_ids : dict
            Mapping from input P_spec identifiers to generated E_0001-style IDs.
        species_list : list of str
            All mechanism species in CSV column order, including unused species.

        Raises
        ------
        ValueError
            If a CAS has inconsistent known MWt values, a (P_spec, CAS)
            pair is duplicated, a retained profile CAS is absent from the mechanism,
            or a species receives contributions from multiple original pollutants
            within one profile.

        Notes
        -----
        Sector components with missing CAS or MWt are retained for weight totals,
        warned about and excluded from factors without renormalization. Rows with
        missing CAS in the mechanism file are discarded. In the sector CSV,
        ``unspecified`` is also parsed as a missing value. Missing configured
        pollutants and weight totals differing from 100 by at least 5 percentage
        points generate log messages, not exceptions. Weights are not normalized.
        Unselected pollutant rows are excluded from validation and factors after
        a warning. Profiles containing only unselected pollutants retain an empty
        conversion mapping; missing selected coverage is checked against emissions.
        Equivalent profiles are merged by exact dictionary equality; generated IDs
        follow first occurrence in the sector CSV. Empty species mappings are removed
        from factors but their names remain in species_list.
        """

        log_message("Obtaining speciation SNAP profiles", level=2)

        column_types = {
            'P_spec': str,
            'CAS': str,
            'Species_name': str,
            'Weight_percentual': precision,
            'MWt': precision,
            'Specie_ori': str}

        # ====== Read the 1st file ======
        # Keep unresolved components for composition totals; unspecified is parsed as missing.
        df1 = read_csv(self.spec_snap_path, usecols=list(column_types.keys()), dtype=column_types,
                       na_values="unspecified")

        self._profile_totals = df1.groupby(['P_spec', 'Specie_ori'])['Weight_percentual'].sum().to_dict()
        # Keep profile IDs even when all their pollutants are unselected.
        input_profile_ids = df1['P_spec'].unique().tolist()
        coverage_profiles = df1.set_index(['P_spec', 'CAS'])
        df1 = self.__process_profiles_and_checks__(df1, pollutants)
        self._warn_missing_profile_pollutants(coverage_profiles, pollutants)

        # Read mechanism coefficients, skipping the units and description rows.
        # Get species names from columns
        species_list = read_csv(self.indi_spec_path, nrows=1).columns.tolist()[2:]
        column_types = {
            'CAS': str,
            'Species_name': str
        }
        for pol_name in species_list:
            column_types[pol_name] = precision

        df2 = read_csv(self.indi_spec_path, skiprows=lambda x: x in [1, 2], usecols=list(column_types.keys()),
                       dtype=column_types)
        df2 = df2.dropna(subset=["CAS"])
        df2.set_index(['CAS'], inplace=True)

        # Check if all CAS in df1 are in df2
        self.__check_CAS_df1_in_df2__(df1, df2)

        # Organize df2 into a dict, chemical mechanism species as 1st level keys,
        # individual pollutants as 2nd level keys
        _, profile_dict = self.__to_dictonary__(df1, df2, species_list)
        spec_ids = input_profile_ids

        # Create profiles for each spec_id by substitute indi pollutants in profile_dict with original pollutants in df1
        profile_dict2 = self.__create_profiles__(df1, spec_ids, profile_dict)

        # Remove duplicates and create new IDs
        duplicates, mapping = self.__remove_duplicates__(profile_dict2)

        # Remove empty species
        for cas in duplicates.keys():
            duplicates[cas] = {key: value for key, value in duplicates[cas].items() if value}

        return duplicates, mapping, species_list

    def _warn_missing_profile_pollutants(self, profiles: DataFrame, pollutants: List[str]) -> None:
        """Warn about nonzero annual emissions without an original-pollutant profile.

        Check sector and point-source inventories independently before P_spec IDs
        are remapped. Only used profiles and nonzero, nonmissing emission records
        are considered. Amounts are summed in input inventory units (t/year),
        before speciation and unit conversion. Missing profile IDs remain subject
        to the existing error in map_emission_p_spec(). No factors are modified.
        """
        coverage = profiles.reset_index().groupby('P_spec')['Specie_ori'].agg(set).to_dict()
        inventories = [(self.sector_info, 'sector inventory'),
                       (self.point_sources, 'point-source inventory')]
        for inventory, inventory_name in inventories:
            if inventory is None:
                continue
            for profile_id, records in inventory.groupby('P_spec'):
                if profile_id not in coverage:
                    continue
                for pollutant in pollutants:
                    if pollutant in coverage[profile_id]:
                        continue
                    emissions = records[pollutant]
                    affected = emissions.loc[emissions.notna() & emissions.ne(0)]
                    if affected.empty:
                        continue
                    message = (
                        f"Speciation profile '{profile_id}' in '{path.basename(self.spec_snap_path)}' "
                        f"does not include original pollutant '{pollutant}'. "
                        f"{affected.sum():.12g} t/year from {len(affected)} {inventory_name} records "
                        "will not contribute to the speciated output.")
                    self._record_issue('warning', 'missing_profile_pollutant', message, 'continue',
                                       'Nonzero original emissions have no conversion factors.',
                                       profile_id=profile_id, original_pollutant=pollutant,
                                       inventory_type=inventory_name, emission_amount_t_year=affected.sum(),
                                       affected_records=len(affected))
                    log_message(message, level=7)

    def __check_if_pollutants_in_speciation__(self, profiles, pollutants) -> None:
        """
        Check pollutant coverage between the profile CSV and configuration.

        Parameters
        ----------
        profiles : DataFrame
            Retained profile rows with a Specie_ori column.
        pollutants : list of str
            Configured original pollutants.

        Notes
        -----
        A configured pollutant absent from the profile CSV only generates a log
        message. Profile pollutants absent from the configuration also generate
        a warning and are subsequently excluded from validation and factors.
        Coverage is checked across the whole CSV, not separately per profile.
        """
        if not all(elem in pollutants for elem in profiles["Specie_ori"].unique()):
            missing_elem = [elem for elem in profiles["Specie_ori"].unique() if elem not in pollutants]
            warn_msg = ("Pollutants " + ", ".join(sorted(missing_elem)) + " in " +
                        path.basename(self.spec_snap_path) + " are not selected in the sector configuration. "
                        "Their profile rows will be excluded from validation and speciation factors.")
            self._record_components(profiles.loc[~profiles['Specie_ori'].isin(pollutants)],
                                    'warning', 'unselected_pollutant', warn_msg, 'exclude_unselected',
                                    'Unrequested pollutant rows are excluded from validation and factors.')
            log_message(warn_msg, level=7)

        if not all(elem in profiles["Specie_ori"].unique() for elem in pollutants):
            missing_elem = [elem for elem in pollutants if elem not in profiles["Specie_ori"].unique()]
            warn_msg = ("Found pollutant " + ",".join(missing_elem) +
                        "required in sector configuration file but not exist in " + path.basename(self.spec_snap_path))
            for pollutant in missing_elem:
                self._record_issue('warning', 'missing_file_pollutant', warn_msg, 'continue',
                                   'Configured pollutant has no profile anywhere in this file.',
                                   original_pollutant=pollutant)
            log_message(warn_msg, level=7)

    def __check_NA_in_profiles__(self, profiles) -> None:
        """
        Warn about selected components excluded from conversion for missing CAS or MWt.

        Rows remain in composition totals. No missing value is imputed and no
        remaining component is renormalized. Other numeric validity checks are
        outside this method's scope.
        """
        unresolved = profiles.loc[profiles['CAS'].isna() | profiles['MWt'].isna()]
        for _, row in unresolved.iterrows():
            missing = [field for field in ['CAS', 'MWt'] if isna(row[field])]
            component = row['Species_name'] if not isna(row['Species_name']) else 'unnamed'
            cas = row['CAS'] if not isna(row['CAS']) else 'missing'
            message = (
                f"Speciation profile '{row['P_spec']}' in '{path.basename(self.spec_snap_path)}': "
                f"component '{component}' (CAS: {cas}) of original pollutant '{row['Specie_ori']}' "
                f"has missing {' and '.join(missing)}. Its weight ({row['Weight_percentual']:.12g}%) "
                "is retained in composition totals but excluded from conversion factors; "
                "remaining weights are not normalized.")
            self._record_components(row.to_frame().T, 'warning', 'missing_component_data', message,
                                    'exclude_component', 'Component weight is not converted; no renormalization.',
                                    excluded_weight_percent=row['Weight_percentual'])
            log_message(message, level=7)

    def __check_CAS_MWt__(self, profiles) -> None:
        """
        Raise ValueError if a CAS has multiple MWt values across retained profiles.
        """
        if not all(profiles[["CAS", "MWt"]].groupby("CAS")["MWt"].nunique() == 1):
            wrong_cas = profiles[["CAS", "MWt"]].groupby("CAS")["MWt"].nunique() == 1
            err_msg = ("Found CAS " + ",".join(wrong_cas[~wrong_cas].index.to_list()) + " in " +
                       path.basename(self.spec_snap_path) + " don't have unique MWt.")
            self._record_components(profiles.loc[profiles['CAS'].isin(wrong_cas[~wrong_cas].index)],
                                    'error', 'inconsistent_molecular_weight', err_msg, 'stop',
                                    'Conflicting molecular weights prevent factor construction.')
            raise ValueError(err_msg)

    def __check_sum_weight_percentual__(self, profiles) -> None:
        """
        Log weight totals outside the accepted range for each (P_spec, Specie_ori).

        Totals with abs(total - 100) >= 5 generate a log message. This method neither
        raises an exception nor normalizes weights. Selected components with missing
        CAS or MWt are included in these totals, even though they cannot be converted.
        """
        tolerance = 5
        tmp = profiles[["P_spec", "Weight_percentual", "Specie_ori"]].groupby(["P_spec", "Specie_ori"]).sum()
        if any(abs(tmp["Weight_percentual"] - 100) >= tolerance):
            warn_msg = ("Found " + str(tmp[abs(tmp["Weight_percentual"] - 100) >= tolerance].index.to_list()) +
                        " in " + path.basename(self.spec_snap_path) + " their components don't sum up to 100.")
            for (profile_id, pollutant), total in tmp['Weight_percentual'].items():
                if abs(total - 100) >= tolerance:
                    self._record_issue('warning', 'weight_total_deviation', warn_msg, 'continue',
                                       'Weights are applied as supplied, without normalization.',
                                       profile_id=profile_id, original_pollutant=pollutant, total_weight_percent=total)
            log_message(warn_msg, level=7)

    def __check_P_spec_CAS_unique__(self, profiles) -> None:
        """
        Raise ValueError for duplicate (P_spec, CAS) index entries.

        The index must already have been set to these two columns.
        """
        if len(profiles.index) != len(profiles.index.unique()):
            repeated_indices = profiles[profiles.index.duplicated(keep='first')].index.tolist()
            err_msg = ("Found repeated (P_spec, CAS) " + str(repeated_indices) + " in " +
                       path.basename(self.spec_snap_path))
            rows = profiles.loc[profiles.index.duplicated(keep=False)].reset_index()
            self._record_components(rows.drop_duplicates(['P_spec', 'CAS']), 'error', 'duplicate_component',
                                    err_msg, 'stop', 'Repeated component IDs make the profile ambiguous.')
            raise ValueError(err_msg)

    def __process_profiles_and_checks__(self, profiles, pollutants) -> DataFrame:
        """
        Filter unselected pollutants, validate retained rows and set their index.

        Parameters
        ----------
        profiles : DataFrame
            All profile rows, including components with missing CAS or MWt.
        pollutants : list of str
            Configured original pollutants.

        Returns
        -------
        DataFrame
            Selected pollutant rows indexed by (P_spec, CAS). Validations may raise
            ValueError; pollutant omissions and weight-total deviations are logged.
        """
        # Check if all the pollutants listed in config file exist in speciation file, and vise versa.
        # Profile pollutants outside the configuration are warned about and excluded.
        # If in config, but not in speciation file --> warning
        self.__check_if_pollutants_in_speciation__(profiles, pollutants)
        # Unselected pollutants must not enter validation or conversion factors.
        profiles = profiles.loc[profiles['Specie_ori'].isin(pollutants)].copy()
        # Validate composition before removing unresolved components from calculation.
        self.__check_sum_weight_percentual__(profiles)
        self.__check_NA_in_profiles__(profiles)
        # Components without CAS have no stable identity; allow separate unresolved rows.
        known_components = profiles.loc[profiles['CAS'].notna()].set_index(['P_spec', 'CAS'])
        self.__check_P_spec_CAS_unique__(known_components)
        profiles = profiles.loc[profiles['CAS'].notna() & profiles['MWt'].notna()].copy()
        self.__check_CAS_MWt__(profiles)
        profiles.set_index(['P_spec', 'CAS'], inplace=True)

        return profiles

    def __check_CAS_df1_in_df2__(self, df1, df2) -> None:
        # Check if all CAS in df1 are in df2
        """
        Raise ValueError if a retained sector-profile CAS is absent from the mechanism.

        Parameters
        ----------
        df1 : DataFrame
            Sector profiles indexed by (P_spec, CAS).
        df2 : DataFrame
            Mechanism coefficients indexed by CAS.
        """
        df1_cas = df1.index.get_level_values(1).tolist()
        df2_cas = df2.index.tolist()
        cas_not_in_df2 = [index for index, elem in enumerate(df1_cas) if elem not in df2_cas]
        if cas_not_in_df2:
            err_msg = "Found " + path.basename(self.spec_snap_path) + " has CAS " + \
                    ' '.join(set([df1_cas[i] for i in cas_not_in_df2])) + " but not in " + \
                    path.basename(self.indi_spec_path) + "."
            self._record_components(df1.iloc[cas_not_in_df2].reset_index(), 'error', 'missing_mechanism_cas',
                                    err_msg, 'stop', 'Component has no entry in the chemical mechanism.',
                                    related_file=self.indi_spec_path)
            raise ValueError(err_msg)

    def __to_dictonary__(self, df1, df2, species_list) -> Tuple[List, Dict]:
        # Organize df2 into a dict, chemical mechanism species as 1st level keys, individual pollutants
        # as 2nd level keys
        """
        Collect input profile IDs and mechanism coefficients by species and CAS.

        Returns
        -------
        tuple
            Profile IDs in first-occurrence order and a {species: {CAS: coefficient}}
            mapping. Zero coefficients are omitted. Missing coefficients are not
            validated here and can propagate into calculated factors.
        """
        spec_ids = df1.reset_index()["P_spec"].unique().tolist()
        profile_dict = {}
        for spec in species_list:
            profile_dict[spec] = {}

        for spec_name in species_list:
            row_with_value = df2.loc[df2[spec_name] != 0, spec_name]
            for ind, cas in enumerate(row_with_value.index):
                profile_dict[spec_name][cas] = row_with_value.iloc[ind]

        return spec_ids, profile_dict

    def __create_profiles__(self, df1, spec_ids, profile_dict) -> Dict:
        """
        Combine component weights with mechanism coefficients for each profile.

        For each retained component, calculate::

            factor = (Weight_percentual / 100) / MWt
                     * mechanism_coefficient * 1000

        Weight_percentual is expressed in percent. MWt is conventionally in g/mol.
        The factor 1000 is fixed in the implementation: output units depend on the inventory units and coefficient
        conventions, including those used for mass-based particle species. Unit
        metadata from the mechanism CSV does not alter this calculation.

        Returns
        -------
        dict
            {profile_id: {species: {original_pollutant: summed_factor}}}.
            Species with no contributing component have an empty mapping.

        Raises
        ------
        ValueError
            If one species receives contributions from more than one original
            pollutant within a profile.
        """
        profile_dict2 = {}
        for key in spec_ids:
            profile_dict2[key] = {}
            for spec_id in profile_dict.keys():
                profile_dict2[key][spec_id] = {}

        for spec_id in profile_dict2.keys():
            if spec_id not in df1.index.get_level_values('P_spec'):
                continue
            df1_id_cas = df1.loc[spec_id].index.tolist()
            for i_specie in profile_dict.items():
                # Only keep the CAS that exist in df1
                used_cas = [elem for index, elem in enumerate(i_specie[1].keys()) if elem in df1_id_cas]
                specie_ori = df1.loc[
                    (spec_id, used_cas), ["Species_name", "Specie_ori", "Weight_percentual", "MWt"]]
                if not specie_ori.empty:
                    if len(specie_ori["Specie_ori"].unique()) > 1:
                        err_msg = "Specie '" + i_specie[0] + "' has individual pollutants '" + \
                                ','.join(specie_ori["Species_name"]) + "' in P_spec '" + spec_id + \
                                "' while they correspond to more than one original pollutant: " + \
                                ','.join(specie_ori["Specie_ori"].unique())
                        self._record_components(specie_ori.reset_index(), 'error', 'multiple_original_pollutants',
                                                err_msg, 'stop', 'Destination species has multiple input pollutants.',
                                                destination_species=i_specie[0], related_file=self.indi_spec_path)
                        raise ValueError(err_msg)
                    # Convert weight percent to a fraction and calculate component factors.
                    specie_ori['Weight_percentual'] = specie_ori.apply(
                        lambda row: (row['Weight_percentual'] /
                                     (100 * row['MWt'])) * i_specie[1][row.name[1]] * 1000, axis=1)
                    # Weight_percentual now holds conversion factors, not percentages.
                    # Unit conventions and the fixed multiplier are documented above.

                    # row['Weight_percentual'] -> %
                    # row['MWt'] ->  Molecular weight g/mol
                    # i_specie[1][row.name[1]] -> individual specie to MECH
                    # 1000 -> M to k
                    # division per 100 to pass from per 100 to per 1

                    specie_ori = specie_ori.groupby("Specie_ori")["Weight_percentual"].sum()
                    profile_dict2[spec_id][i_specie[0]] = specie_ori.to_dict()

        return profile_dict2

    def __remove_duplicates__(self, profile_dict2) -> Tuple[Dict, Dict]:
        # Remove duplicates and create new IDs
        """
        Merge profiles with exactly equal factor dictionaries.

        Returns
        -------
        tuple of dict
            Unique factors keyed by generated E_0001-style IDs and the mapping from
            original IDs to generated IDs. IDs follow input iteration order; floating
            point values are compared exactly, without a numerical tolerance.
        """
        duplicates = {}
        mapping = {}
        category_counter = 1

        for key, value in profile_dict2.items():
            category_found = False
            # Check if the value already exists in duplicates
            for category, existing_value in duplicates.items():
                if value == existing_value:
                    mapping[key] = category
                    category_found = True
                    break
            # If value is not found in duplicates, create a new category
            if not category_found:
                category_key = f'E_{str(category_counter).zfill(4)}'
                duplicates[category_key] = value
                mapping[key] = category_key
                category_counter += 1

        return duplicates, mapping

    def map_emission_p_spec(self, dataframe, mapping: dict, sector_name: str) -> DataFrame:
        """
        Map column "P_spec" in sector_info or LPS data frame with new speciation profile keys.

        Parameters
        ----------
        dataframe: DataFrame
            SEIE or LPS data.
        mapping: dict
            Dictionary containing the mapping between original P_spec IDs and new profile IDs.
        sector_name: str
            The name of the sector being processed.

        Returns
        -------
        DataFrame
            Returns a DataFrame with the same structure as the input DataFrame, but with the "P_spec" column updated to
            use the new profile IDs from the provided mapping.

        Raises
        ------
        ValueError
            If any P_spec in the emissions DataFrame is not present in the provided mapping.

        Notes
        -----
        Updates the input DataFrame in place and returns that same object.
        """

        # Validate profile coverage for either sector or point-source emissions.
        if not set(dataframe['P_spec']).issubset(set(mapping.keys())):
            missing_id = set(dataframe['P_spec']) - set(mapping.keys())
            # Use column name to distinguish if it is SEIE or LPS
            column_to_recognize = {
                'is_point_source': 'SEIE',
                'Code': 'LPS',
            }
            identifying_columns = dataframe.columns.intersection(column_to_recognize.keys())
            data_name = (column_to_recognize[identifying_columns[0]] if len(identifying_columns)
                         else 'data (unrecognized file)')
            missing_profiles = ",".join(sorted(str(profile_id) for profile_id in missing_id))
            err_msg = ("Found P_spec " + missing_profiles + " in " + data_name +
                       " but not in speciation_" + sector_name + ".csv.")
            inventory_type = {'SEIE': 'sector inventory', 'LPS': 'point-source inventory'}.get(data_name, data_name)
            if dataframe is self.sector_info:
                inventory_type = 'sector inventory'
            elif dataframe is self.point_sources:
                inventory_type = 'point-source inventory'
            for profile_id in sorted(missing_id, key=str):
                self._record_issue('error', 'missing_inventory_profile', err_msg, 'stop',
                                   'Inventory profile has no conversion mapping.', profile_id=profile_id,
                                   inventory_type=inventory_type)
            raise ValueError(err_msg)

        dataframe['P_spec'] = dataframe['P_spec'].map(mapping)

        return dataframe


def speciate(emissions: DataFrame, speciation_factors: dict, species_list: list,
             original_pollutants: list) -> DataFrame:
    """
    Apply profile factors to sector or point-source emissions.

    Parameters
    ----------
    emissions : DataFrame
        Inventory with P_spec and original pollutant columns. Profile IDs must
        already match speciation_factors, normally through initialize().
    speciation_factors : dict
        {profile_id: {species: {original_pollutant: factor}}}. Each nonempty
        species mapping must contain exactly one original pollutant.
    species_list : list of str
        All species columns to create, including those without contributions.
    original_pollutants : list of str
        Input columns to remove after calculating species.

    Returns
    -------
    DataFrame
        Inventory with species columns replacing original_pollutants and P_spec.
        Other columns and index labels are retained, but row order may change.

    Notes
    -----
    Species are initialized to zero on the input object before calculations;
    this function therefore partially mutates its input. Species names must not
    collide with original pollutant names or metadata columns: initialization
    would overwrite them, and original_pollutants are removed at the end.
    Species without factors and rows with unmatched profiles remain zero.
    Original pollutant columns are removed even when they have no conversion
    factor. Profile coverage is not validated here; the initialization phase
    normally validates P_spec identifiers. Factors are applied as supplied,
    without consulting unit metadata or performing additional unit conversion.
    """

    # Create empty columns for chemical mechanism species
    emissions[species_list] = 0.0

    # Calculate each species from original pollutants
    for spec_id, species in speciation_factors.items():
        rows_to_modify = DataFrame(emissions.loc[emissions["P_spec"] == spec_id],
                                   columns=emissions.columns)
        for spec, factor in species.items():
            num = [float(val) for val in factor.values()][0]
            rows_to_modify[spec] = rows_to_modify[factor.keys()] * num
        emissions = concat([emissions.loc[emissions["P_spec"] != spec_id], rows_to_modify])
    # Remove original pollutants and P_spec
    emissions = emissions.drop(columns=original_pollutants)
    emissions = emissions.drop(columns="P_spec")

    return emissions
