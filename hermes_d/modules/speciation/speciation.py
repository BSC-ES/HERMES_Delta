from typing import Dict, Tuple, List
from pandas import DataFrame, read_csv, concat
from numpy import isnan
from os import path, makedirs
import hermes_d
from hermes_d.config import precision, log_message


class SpeciationDelta(object):
    def __init__(self, spec_snap_path: str, indi_spec_path: str, spec_aux_path: str,
                 sector_info: DataFrame,
                 point_sources: DataFrame = None):
        """
        Initialize the SpeciationDelta class by obtaining the speciation mapping.

        Parameters
        ----------
        sector_info: DataFrame
            A DataFrame containing sector information, including a "P_spec" column.
        point_sources: DataFrame, optional
            A DataFrame containing point source information, including a "P_spec" column. Default is None.
        """
        self.spec_snap_path = spec_snap_path
        self.indi_spec_path = indi_spec_path
        self.spec_aux_path = spec_aux_path

        self.speciation_mapping = None
        self.species_list = None

        self.sector_info = sector_info
        self.point_sources = point_sources

    def initialize(self, pollutants, sector_name) -> Tuple[DataFrame, DataFrame, List, Dict]:
        """
        Initialize the speciation process by obtaining the speciation mapping and updating the sector_info and
        point_sources DataFrames with new speciation profile IDs.
        Parameters
        ----------
        pollutants: list
            A list of pollutants provided in Sector config file.
        sector_name: str
            The name of the sector being processed.

        Returns
        -------
        Tuple[DataFrame, DataFrame, List, Dict]
            Updated sector_info DataFrame, updated point_sources DataFrame (if provided), list of species
            in the chemical mechanism, and the speciation mapping dictionary.
        """
        self.speciation_mapping, mapping_p_spec, self.species_list = self.get_speciation_mapping(pollutants)

        if hermes_d.DEBUG:
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

    # noinspection DuplicatedCode
    def get_speciation_mapping(self, pollutants) -> Tuple[Dict, Dict, List]:
        """
        Obtain the speciation mapping between original pollutants and chemical mechanism species by combining
        information from speciation_SNAPXX.csv and individual_speciation_<chemical_mechanism>.csv.

        Parameters
        ----------
        pollutants: list
            A list of pollutants provided in Sector config file.

        Returns
        -------
        Tuple[Dict, Dict, List]
            First item is a dictionary where each key is a new profile ID and the value is a dictionary mapping chemical
            mechanism species to their corresponding original pollutants.
            Second item is a dictionary mapping original P_spec IDs to new profile IDs.
            Third item is a list of the species names in the chemical mechanism.

        Raises
        ------
        ValueError
            If CAS in spec_snap_path don't have unique MWt, or Weight_percentual doesn't sum up to 100.
            If CAS in spec_snap_path are not found in indi_spec_path.
        """

        log_message("Obtaining speciation SNAP profiles", level=2)

        column_types = {
            'P_spec': str,
            'CAS': str,
            'Species_name': str,
            'TOG_to_VOC_ratio': precision,
            'Weight_percentual': precision,
            'MWt': precision,
            'Specie_ori': str}

        # ====== Read the 1st file ======
        # TODO: What values should be NA (unspecified, unspeciated, unknown, other, ___?)
        df1 = read_csv(self.spec_snap_path, usecols=list(column_types.keys()), dtype=column_types,
                       na_values="unspecified")
        df1 = df1.dropna(subset=["CAS"])

        df1 = self.__process_profiles_and_checks__(df1, pollutants)

        # TODO: is profiles the right name?

        # ====== Read the 2nd file ======
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
        spec_ids, profile_dict = self.__to_dictonary__(df1, df2, species_list)

        # Create profiles for each spec_id by substitute indi pollutants in profile_dict with original pollutants in df1
        profile_dict2 = self.__create_profiles__(df1, spec_ids, profile_dict)

        # Remove duplicates and create new IDs
        duplicates, mapping = self.__remove_duplicates__(profile_dict2)

        # Remove empty species
        for cas in duplicates.keys():
            duplicates[cas] = {key: value for key, value in duplicates[cas].items() if value}

        return duplicates, mapping, species_list

    def __check_if_pollutants_in_speciation__(self, profiles, pollutants) -> None:
        """
        Check if all pollutants in the provided list exist in the speciation profiles.

        Parameters
        ----------
        profiles: DataFrame
            The DataFrame containing speciation profiles with a "Specie_ori" column.
        pollutants: list
            A list of pollutants to check against the speciation profiles.

        Raises
        ------
        ValueError
            If any pollutant in the provided list does not exist in the speciation profiles.
        """
        if not all(elem in pollutants for elem in profiles["Specie_ori"].unique()):
            missing_elem = [elem for elem in profiles["Specie_ori"].unique() if elem not in pollutants]
            err_msg = "Pollutant " + ",".join(missing_elem) + " exist in " + path.basename(self.spec_snap_path) + \
                      " but not listed in sector configuration file."
            raise ValueError(err_msg)

        if not all(elem in profiles["Specie_ori"].unique() for elem in pollutants):
            missing_elem = [elem for elem in pollutants if elem not in profiles["Specie_ori"].unique()]
            warn_msg = ("Found pollutant " + ",".join(missing_elem) +
                        "required in sector configuration file but not exist in " + path.basename(self.spec_snap_path))
            log_message(warn_msg, level=7)

    def __check_NA_in_profiles__(self, profiles) -> None:
        """ Check if there are any NA values in the MWt or TOG_to_VOC_ratio columns of the profiles DataFrame."""
        if any(isnan(profiles["MWt"])):
            err_msg = ("MWt of the pollutant " + ",".join(profiles["CAS"][isnan(profiles["MWt"])]) + " is NA in " +
                       path.basename(self.spec_snap_path))
            raise ValueError(err_msg)

        if any(isnan(profiles["TOG_to_VOC_ratio"])):
            err_msg = ("TOG_to_VOC_ratio of the pollutant " +
                       ",".join(profiles["CAS"][isnan(profiles["TOG_to_VOC_ratio"])]) +
                       "is NA in " + path.basename(self.spec_snap_path))
            raise ValueError(err_msg)

    def __check_CAS_MWt__(self, profiles) -> None:
        """ Check if each CAS has a unique corresponding MWt in the profiles DataFrame."""
        if not all(profiles[["CAS", "MWt"]].groupby("CAS")["MWt"].nunique() == 1):
            wrong_cas = profiles[["CAS", "MWt"]].groupby("CAS")["MWt"].nunique() == 1
            err_msg = ("Found CAS " + ",".join(wrong_cas[~wrong_cas].index.to_list()) + " in " +
                       path.basename(self.spec_snap_path) + " don't have unique MWt.")
            raise ValueError(err_msg)

    def __check_sum_weight_percentual__(self, profiles) -> None:
        """ Check if the sum of Weight_percentual for each P_spec in the profiles DataFrame equals 100."""
        tolerance = 5
        tmp = profiles[["P_spec", "Weight_percentual", "Specie_ori"]].groupby(["P_spec", "Specie_ori"]).sum()
        if any(abs(tmp["Weight_percentual"] - 100) >= tolerance):
            warn_msg = ("Found " + str(tmp[abs(tmp["Weight_percentual"] - 100) >= tolerance].index.to_list()) +
                        " in " + path.basename(self.spec_snap_path) + " their components don't sum up to 100.")
            log_message(warn_msg, level=7)

    def __check_P_spec_CAS_unique__(self, profiles) -> None:
        """ Check if all (P_spec, CAS) pairs in the profiles DataFrame are unique."""
        if len(profiles.index) != len(profiles.index.unique()):
            repeated_indices = profiles[profiles.index.duplicated(keep='first')].index.tolist()
            err_msg = ("Found repeated (P_spec, CAS) " + str(repeated_indices) + " in " +
                       path.basename(self.spec_snap_path))
            raise ValueError(err_msg)

    def __process_profiles_and_checks__(self, profiles, pollutants) -> None:
        """
        Process the profiles DataFrame and perform necessary checks.

        Parameters
        ----------
        profiles: DataFrame
            The DataFrame containing speciation profiles.
        pollutants: list
            A list of pollutants to check against the speciation profiles.
        """
        # Check if all the pollutants listed in config file exist in speciation file, and vise versa.
        # If not in config, but in speciation file --> error
        # If in config, but not in speciation file --> warning
        self.__check_if_pollutants_in_speciation__(profiles, pollutants)
        # Check if any MWt is NA or TOG_to_VOC_ratio is NA
        self.__check_NA_in_profiles__(profiles)
        # Check if each CAS has the same corresponding MWt
        self.__check_CAS_MWt__(profiles)
        # Check if the sum of Weight_percentual is 100. If not, warning
        self.__check_sum_weight_percentual__(profiles)

        profiles.set_index(['P_spec', 'CAS'], inplace=True)

        # Check if all (P_spec, CAS) is unique
        self.__check_P_spec_CAS_unique__(profiles)

        return profiles

    def __check_CAS_df1_in_df2__(self, df1, df2) -> None:
        # Check if all CAS in df1 are in df2
        df1_cas = df1.index.get_level_values(1).tolist()
        df2_cas = df2.index.tolist()
        cas_not_in_df2 = [index for index, elem in enumerate(df1_cas) if elem not in df2_cas]
        if cas_not_in_df2:
            err_msg = "Found " + path.basename(self.spec_snap_path) + " has CAS " + \
                    ' '.join(set([df1_cas[i] for i in cas_not_in_df2])) + " but not in " + \
                    path.basename(self.indi_spec_path) + "."
            raise ValueError(err_msg)

    def __to_dictonary__(self, df1, df2, species_list) -> Dict:
        # Organize df2 into a dict, chemical mechanism species as 1st level keys, individual pollutants
        # as 2nd level keys
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
        profile_dict2 = {}
        for key in spec_ids:
            profile_dict2[key] = {}
            for spec_id in profile_dict.keys():
                profile_dict2[key][spec_id] = {}

        for spec_id in profile_dict2.keys():
            df1_id_cas = df1.loc[spec_id].index.tolist()
            for i_specie in profile_dict.items():
                # Only keep the CAS that exist in df1
                used_cas = [elem for index, elem in enumerate(i_specie[1].keys()) if elem in df1_id_cas]
                specie_ori = df1.loc[
                    (spec_id, used_cas), ["Species_name", "Specie_ori", "TOG_to_VOC_ratio", "Weight_percentual", "MWt"]]
                if not specie_ori.empty:
                    if len(specie_ori["Specie_ori"].unique()) > 1:
                        err_msg = "Specie '" + i_specie[0] + "' has individual pollutants '" + \
                                ','.join(specie_ori["Species_name"]) + "' in P_spec '" + spec_id + \
                                "' while they correspond to more than one original pollutant: " + \
                                ','.join(specie_ori["Specie_ori"].unique())
                        raise ValueError(err_msg)
                    # division per 100 to pass from per 100 to per 1
                    specie_ori['Weight_percentual'] = specie_ori.apply(
                        lambda row: (row['TOG_to_VOC_ratio'] * row['Weight_percentual'] /
                                     (100 * row['MWt'])) * i_specie[1][row.name[1]] * 1000, axis=1)
                    # row['TOG_to_VOC_ratio'] -> %
                    # row['Weight_percentual'] -> %
                    # row['MWt'] ->  Molecular weight g/mol
                    # i_specie[1][row.name[1]] -> individual specie to MECH
                    # 1000 -> M to k

                    specie_ori = specie_ori.groupby("Specie_ori")["Weight_percentual"].sum()
                    profile_dict2[spec_id][i_specie[0]] = specie_ori.to_dict()

        return profile_dict2

    def __remove_duplicates__(self, profile_dict2) -> None:
        # Remove duplicates and create new IDs
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
        """

        # Check: if all P_spec in SEIE are in speciation_SNAPxx.csv
        if not set(dataframe['P_spec']).issubset(set(mapping.keys())):
            missing_id = set(dataframe['P_spec']) - set(mapping.keys())
            # Use column name to distinguish if it is SEIE or LPS
            column_to_recognize = {
                'is_point_source': 'SEIE',
                'Code': 'LPS',
            }
            data_name = column_to_recognize.get(
                dataframe.columns.intersection(column_to_recognize.keys()).tolist()[0], 'data (unrecognized file)')
            err_msg = ("Found P_spec " + ",".join(missing_id) + " in " + data_name +
                       " but not in speciation_" + sector_name + ".csv.")
            raise ValueError(err_msg)

        dataframe['P_spec'] = dataframe['P_spec'].map(mapping)

        return dataframe


def speciate(emissions: DataFrame, speciation_factors: dict, species_list: list,
             original_pollutants: list) -> DataFrame:
    """
    Convert original pollutants to chemical mechanism species in SEIE data frame.

    Parameters
    ----------
    emissions: DataFrame
        SEIE or LPS data.
    speciation_factors: dict
        The factors to convert original pollutants to chemical mechanism species for each profile
    species_list: list
        The species name.
    original_pollutants: list
        The list of original pollutants

    Returns
    -------
    DataFrame
        Returns a DataFrame with the same structure as the input DataFrame, but with the original pollutants replaced
        with chemical mechanism species. The "P_spec" column is also removed.

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
