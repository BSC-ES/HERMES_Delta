from typing import Dict, Tuple, List, Optional
from warnings import warn
from pandas import DataFrame, Series, date_range, MultiIndex, read_csv, to_datetime, concat
from pytz import timezone
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange, isleap
from numpy import allclose, isclose, array
from yaml import safe_load as load_yaml
from hermes_d.config import precision, log_message
from hermes_d.utilities import dict_to_ordered_str, str_to_dict
from holidays import country_holidays


class TemporalDelta(object):
    """
    A class to handle temporal scaling and profile adjustments for emissions data,
    supporting conversions across various temporal profiles (monthly, weekly, daily, and hourly)
    and applying adjustments based on regional time zones.

    This class provides functionalities to distribute annual emissions into hourly values
    by utilizing specific temporal profiles and adjusting for time zone differences.

    Attributes
    ----------
    seie : DataFrame
        Table containing SEIE emissions data. Columns and index requirements should
        be specified based on the SEIE data structure.

    point_sources : DataFrame or None
        Table containing point source emissions data, or None if not applicable.
        Columns and index requirements should be specified based on the point source data structure.

    profiles_paths : dict
        Dictionary containing file paths to profile data (monthly, weekly, hourly, etc.).
        Expected keys and corresponding file paths should be detailed according to usage.

    time_step_list : list of datetime
        List of UTC datetime objects representing each hourly timestep in the simulation period.

    holidays : bool
        Specifies if regional holidays (NUTS2) should be treated as Sunday profiles.
        This is used to adjust the weekly profile for regions where holidays are treated as non-working days.

    daily_consistency : bool
        Boolean indicating whether to raise an error if the temporal resolution is
        daily and the sum of the temporal profiles does not match the leap year or non-leap year
        status of the simulated year.
    """
    def __init__(self, seie: DataFrame, point_sources: Optional[DataFrame], profiles_paths: dict, time_step_list: list,
                 holidays: bool = False, daily_consistency: bool = True):
        """
        Initializes the TemporalDelta class with SEIE data, point sources, profile paths,
        a list of timesteps, and holiday treatment options.

        Parameters
        ----------
        seie : DataFrame
            Table containing SEIE emissions data. The structure and required columns
            should align with the specific SEIE dataset format.

        point_sources : DataFrame or None
            Table containing point source emissions data, or None if not applicable.
            The structure and required columns should align with the specific point source dataset format.

        profiles_paths : dict
            Dictionary that specifies paths to the temporal profile data files.
            Expected keys should include those necessary for accessing monthly, weekly,
            and hourly profile data.

        time_step_list : list of datetime
            A list of datetime objects representing each timestep (in UTC) to be processed
            in the simulation period.

        holidays : bool
            Boolean indicating whether to apply special treatment for regional holidays
            (NUTS2 level), assigning a Sunday profile for holiday dates.

        daily_consistency : bool
            Boolean indicating whether to raise an error if the temporal resolution is
            daily and the sum of the temporal profiles does not match the leap year or non-leap year
            status of the simulated year.
        """
        self.seie = seie

        if self.seie is not None:
            if self.seie.empty:
                raise ValueError(f"No data found in SEIE DataFrame. Please check the input data.")

        self.point_sources = point_sources
        self.profiles_paths = profiles_paths
        self.time_step_list = time_step_list
        if time_step_list is not None:
            self.set_inconsistent_time_steps(time_step_list)
        else:
            self.inconsistent_time_steps = []
        self.daily_consistency = daily_consistency
        self.isleap_profiles = None

        self.holidays = holidays
        if self.holidays:
            self.holiday_regions = self.__load_nuts2_holidays_region__(self.profiles_paths['nuts2_info'])
        else:
            self.holiday_regions = None

    def set_inconsistent_time_steps(self, time_step_list):
        self.inconsistent_time_steps = [None for _ in time_step_list]
        # None if dont know yet, True if inconsistent, False if consistent

    def initialize(self):
        """
        Initializes and prepares temporal profiles for SEIE and point source emissions data by
        validating temporal profile types, mapping emissions data to profiles, and calculating
        hourly factors based on the specified temporal profile type.

        This method performs several checks and initializations:
        1. Identifies and verifies the temporal profile types for SEIE and point source emissions.
        2. Retrieves the relevant temporal profiles based on the profile type.
        3. Maps the SEIE and point source emissions data to the respective profiles.
        4. Combines and normalizes profiles, calculating final hourly factors.

        Returns
        -------
        Tuple[DataFrame, DataFrame or None, DataFrame]
            - Updated SEIE DataFrame with mapped temporal profiles.
            - Updated point sources DataFrame with mapped temporal profiles, or None if no point sources are present.
            - DataFrame containing calculated hourly factors based on the specified temporal profile type.

        Raises
        ------
        ValueError
            If the temporal profile types for SEIE and point sources are inconsistent.

        NotImplementedError
            If an unsupported temporal profile type is encountered.
        """
        temp_prof_types = {"day": ["daily", "hourly"],
                           "month_week": ["monthly", "weekly", "hourly"]}
        data = {}
        temporal_factors_dict = {}

        for temp_prof_type, profile_types in temp_prof_types.items():
            df_seie = self.seie
            df_point_sources = self.point_sources

            # extract only the needed part of the dataframe, and carry out the checks.
            seie, point_sources = self._split_data_by_profile(df_seie, df_point_sources, temp_prof_type)
            if seie.empty:
                log_message(f"No data found for temporal profile type '{temp_prof_type}'. Skipping this profile type.")
                data[temp_prof_type] = {"seie": DataFrame(), "point_sources": None}
                temporal_factors_dict[temp_prof_type] = DataFrame()
                continue  # If no data for this profile type, skip to the next

            # Get temporal profiles, remove duplicates, and create a mapping from old index to new index
            seie, point_sources, temporal_profiles = self._obtain_profiles(profile_types, seie, point_sources)

            # Get final temporal profile and merge profiles
            seie, temp_profile_mapping = self.__get_final_temp_profile__(seie, temp_prof_type)

            if point_sources is not None:
                point_sources, temp_profile_mapping_ps = self.__get_final_temp_profile__(
                    point_sources, temp_prof_type)
                temp_profile_mapping, point_sources = self.__unify_seie_lps_mapping__(
                    temp_profile_mapping, temp_profile_mapping_ps, point_sources)

            if temp_prof_type == 'month_week':
                temporal_factors = self.__calculate_yr_to_hr_profiles__(temporal_profiles, temp_profile_mapping)
            elif temp_prof_type == 'day':
                temporal_factors = self.__calculate_yr_to_hr_profiles__(
                    temporal_profiles, temp_profile_mapping, daily=True)
            else:
                temporal_factors = None

            data[temp_prof_type] = {"seie": seie,
                                    "point_sources": point_sources}

            temporal_factors_dict[temp_prof_type] = temporal_factors

        self.seie = concat([data["day"]["seie"], data["month_week"]["seie"]])
        if data["day"]["point_sources"] is not None or data["month_week"]["point_sources"] is not None:
            self.point_sources = concat([data["day"]["point_sources"], data["month_week"]["point_sources"]])

        # concat temporal factors df, and change the index
        temporal_factors_combined = concat([temporal_factors_dict['day'], temporal_factors_dict['month_week']], axis=0)
        temporal_factors_combined.index = [f"T_{i:04d}" for i in range(1, len(temporal_factors_combined) + 1)]

        return self.seie, self.point_sources, temporal_factors_combined

    def _split_data_by_profile(self, df_seie: DataFrame, df_point_sources: DataFrame, temp_prof_type: str):
        """
        Filters and validates SEIE and point source DataFrames based on the specified temporal profile type.

        Parameters
        ----------
        df_seie : DataFrame
            The SEIE (Spatial Emission Inventory Entities) DataFrame to filter and validate.

        df_point_sources : DataFrame
            The point sources DataFrame to filter and validate. Can be None.

        temp_prof_type : str
            The expected temporal profile type. Must be one of:
            - "day": for daily profiles (uses 'P_day' column).
            - "month_week": for monthly or weekly profiles (uses 'P_month' and/or 'P_week' columns).

        Returns
        -------
        Tuple[DataFrame, DataFrame or None]
            A tuple containing the filtered SEIE and point source DataFrames, respectively. The point sources
            DataFrame may be None if no data is present or was passed in.

        Raises
        ------
        ValueError
            If `temp_prof_type` is not one of the expected values ("day", "month_week").
            If any filtered DataFrame does not conform to the expected temporal profile type, as determined
            by `__find_temp_prof_type__`.

        """
        if temp_prof_type == "day":
            seie = df_seie[df_seie['P_day'].str.len() > 0]
            if not seie.empty:
                if self.__find_temp_prof_type__(seie) != "day":
                    raise ValueError("Some checks not passed for seie df with daily profiles.")
            if df_point_sources is not None:
                point_sources = df_point_sources[df_point_sources['P_day'].str.len() > 0]
                if not point_sources.empty:
                    if self.__find_temp_prof_type__(point_sources) != "day":
                        raise ValueError("Some checks not passed for point source df with daily profiles.")
            else:
                point_sources = None

        elif temp_prof_type == "month_week":
            seie = df_seie[(df_seie['P_month'].str.len() > 0) | (df_seie['P_week'].str.len() > 0)]
            if not seie.empty:
                if self.__find_temp_prof_type__(seie) != "month_week":
                    raise ValueError("Some checks not passed for seie df with month_week profiles.")
            if df_point_sources is not None:
                point_sources = df_point_sources[
                    (df_point_sources['P_month'].str.len() > 0) | (df_point_sources['P_week'].str.len() > 0)]
                if not point_sources.empty:
                    if self.__find_temp_prof_type__(point_sources) != "month_week":
                        raise ValueError("Some checks not passed for point source df with month_week profiles.")
            else:
                point_sources = None

        else:
            raise ValueError("invalid temporal profile type. Must be 'day' or 'month_week'.")

        return seie, point_sources

    def _obtain_profiles(self, profile_types: list, seie, point_sources):
        """
        Obtains the temporal profiles for SEIE and point sources, and updates the emission tables to
        contain the unique profile index.

        Args
        ----
        profile_types : list
            List of strings indicating the types of profiles to obtain (e.g., 'monthly', 'weekly', 'hourly').
        seie:

        point_sources:

        Returns
        -------
        Tuple[DataFrame, DataFrame or None, DataFrame]
            - SEIE DataFrame with mapped temporal profiles.
            - Point sources DataFrame with mapped temporal profiles, or None if not applicable.
            - Dictionary containing the temporal profiles for each type (e.g., 'monthly', 'weekly', 'hourly').
        """
        temporal_profiles = {}
        for i_freq in profile_types:
            temporal_profiles[i_freq], mapping_temporal_ids = self.__get_temporal_mapping__(i_freq)

            # Update emission tables of both seie and point sources
            seie = self.__map_emission_p_temp__(seie, mapping_temporal_ids, i_freq)
            if point_sources is not None:
                point_sources = self.__map_emission_p_temp__(point_sources, mapping_temporal_ids, i_freq)

        return seie, point_sources, temporal_profiles

    @staticmethod
    def __find_temp_prof_type__(emissions: DataFrame) -> str:
        """
        Determine the temporal profile type to use for disaggregating annual emissions to hourly emissions.

        The method can either use the columns "P_month", "P_week", and "P_hour" to disaggregate emissions
        through months, weeks, and hours, or use "P_day" and "P_hour" to disaggregate directly from days
        to hours.

        This function checks for incompatible combinations of temporal profile columns and ensures that
        the required columns are filled based on the chosen disaggregation method.

        Parameters
        ----------
        emissions : DataFrame
            The DataFrame containing the emission data. It may have an Index or a MultiIndex.

        Returns
        -------
        str
            The temporal profile type to use. It can be either 'month_week' (if using "P_month", "P_week", and "P_hour")
            or 'day' (if using "P_day" and "P_hour").

        Raises
        ------
        IndexError
            If any row has values in "P_month", "P_week", and "P_day", indicating that both methodologies are being
            applied simultaneously, which is not allowed. The error message includes the relevant index names.

        ValueError
            If any row has missing values in "P_hour". The error message includes the relevant index names.

            If any row has values in "P_month" but is missing values in "P_week", or vice versa.
            The error message includes the relevant index names.
        """

        def format_index(idx):
            if isinstance(emissions.index, MultiIndex):
                index_names = emissions.index.names
                return ", ".join([f"{index_names[i]} {idx[i]}" for i in range(len(index_names))])
            else:
                return f"{emissions.index.name} {idx}"

        # Check for invalid combination: having values in P_month, P_week, and P_day at the same time
        invalid_combination_indices = emissions[
            emissions[['P_month', 'P_week', 'P_day']].notnull().all(axis=1)
        ].index.tolist()

        if invalid_combination_indices:
            invalid_combinations_str = ", ".join([format_index(idx) for idx in invalid_combination_indices])
            raise IndexError(
                f"Rows at {invalid_combinations_str} cannot have values in both 'P_month', 'P_week', and 'P_day'.")

        # Check for missing values in P_hour
        missing_hour_indices = emissions[emissions['P_hour'].isnull()].index.tolist()

        if missing_hour_indices:
            missing_hours_str = ", ".join([format_index(idx) for idx in missing_hour_indices])
            raise ValueError(f"Rows at {missing_hours_str} are missing values in 'P_hour'.")

        # Check for cases where P_month has values but P_week does not, or vice versa
        invalid_month_week_indices = emissions[
            (emissions['P_month'].notnull() & emissions['P_week'].isnull()) |
            (emissions['P_week'].notnull() & emissions['P_month'].isnull())
            ].index.tolist()

        if invalid_month_week_indices:
            invalid_month_week_str = ", ".join([format_index(idx) for idx in invalid_month_week_indices])
            raise ValueError(f"Rows at {invalid_month_week_str} must have both 'P_month' and 'P_week' "
                             f"values or neither.")

        # Determine which temporal profile method to use
        if emissions[['P_month', 'P_week']].notnull().all().all():
            return 'month_week'
        elif emissions['P_day'].notnull().all():
            return 'day'
        else:
            raise ValueError(
                "All rows must have either 'P_month'/'P_week' or 'P_day' values, "
                "and all rows must have 'P_hour' values.")

    @staticmethod
    def __parse_profile_string__(profile_str: str) -> Dict[str, str]:
        """
        Parse a profile string such as "weekday=H_weekday, weekend=H_weekend"
        and return a dictionary with keys for 'weekday', 'weekend', or 'default' mapping
        to their corresponding profiles. If the key is not valid, raise an error.

        Parameters
        ----------
        profile_str : str
            The profile string that may contain multiple profiles separated by commas.
            If a single profile is provided (e.g., "H_default"), it is automatically assigned to the 'default' key.

        Returns
        -------
        dict
            A dictionary with the corresponding profiles.
            Example: {'weekday': 'H_weekday', 'weekend': 'H_weekend', 'default': 'H_default'}

        Raises
        ------
        ValueError
            If the profile contains invalid keys.

        Allowed Keys
        ------------
        - 'weekday': Applies to weekdays (Monday to Friday).
        - 'weekend': Applies to weekends (Saturday and Sunday).
        - 'mon', 'tue', 'wed', 'thu', 'fri': Specific days of the week (Monday to Friday).
        - 'sat', 'sun': Specific weekend days (Saturday and Sunday).
        - 'default': Applied when no specific days are defined, or when only a single profile is passed
                     (e.g., "H_default").

        Example
        -------
        >>> TemporalDelta().__parse_profile_string__("weekday=H_weekday, weekend=H_weekend")
        {'weekday': 'H_weekday', 'weekend': 'H_weekend'}

        >>> TemporalDelta().__parse_profile_string__("H_default")
        {'default': 'H_default'}
        """
        # Define allowed keys
        allowed_keys = {'weekday', 'weekend', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'default'}

        profiles = {}

        # If the profile string does not contain an '=', assume it's a single default profile
        if '=' not in profile_str:
            profiles['default'] = profile_str.strip()
            return profiles

        # Split the string by commas and iterate over the parts
        for part in profile_str.split(","):
            key, value = part.split("=")
            key = key.strip().lower()  # Convert key to lowercase
            value = value.strip()

            # Check if the key is valid
            if key not in allowed_keys:
                raise ValueError(f"Invalid key '{key}' in profile string. Allowed keys are: {', '.join(allowed_keys)}")

            # Add to the profiles dictionary
            profiles[key] = value

        # Checks
        keys = profiles.keys()
        if "weekday" in keys and not ("weekend" in keys or "default" in keys):
            raise ValueError("Hourly profile for 'weekday' is present but 'weekend' or 'default' is missing.")

        if "weekend" in keys and not ("weekday" in keys or "default" in keys):
            raise ValueError("Hourly profile for 'weekend' is present but 'weekday' or 'default' is missing.")

        return profiles

    @staticmethod
    def isleap_profiles_check(num_cols):
        """ Checks if the daily profiles df is for leap years or not"""
        if num_cols == 366:
            return True
        else:
            return False

    def __get_temporal_mapping__(self, temp_res: str) -> Tuple[DataFrame, Dict]:
        """
        Obtain the temporal profile for emissions and organize them into unique profiles.

        This function reads temporal profiles from a CSV file and maps them to the sector information
        based on the 'P_month', 'P_week', 'P_day', or 'P_hour' column. It creates a DataFrame with unique rows and
        renames the IDs. It also creates a mapping from old index to new index.


        Parameters
        ----------
        temp_res : str
            The temporal resolution of the profile. Can be 'hourly', 'daily', 'weekly', or 'monthly'.

        Returns
        -----
        df_unique : DataFrame
            A DataFrame with unique rows and renamed IDs.
        id_mapping : Dict
            A dictionary mapping the old index to the new index for unique rows.

        Raises
        ------
        ValueError
            If the factors in one row do not sum to the correct number (24.0 for 'hourly', 365/366 for 'daily',
            7.0 for 'weekly', or 12.0 for 'monthly'), a ValueError is raised with a message indicating the old index of
            the row.
        """
        log_message("Obtaining {0} temporal SNAP profiles".format(temp_res), level=2)

        # ====== Read the profile ======
        freq_list = {"hourly": "Hourly_profiles", "daily": "Daily_profiles", "weekly": "Weekly_profiles",
                     "monthly": "Monthly_profiles"}
        path = self.profiles_paths[freq_list[temp_res]]
        df = read_csv(path)
        # Set the column type
        df[df.columns[0]] = df[df.columns[0]].astype(str)
        for col in df.columns[1:]:
            try:
                df[col] = df[col].astype(precision)
            except ValueError as e:
                raise ValueError(f"Error processing {col} column as {precision} in the {path}")
        df.set_index(df.columns[0], inplace=True)

        df_unique = df.drop_duplicates().reset_index(drop=True)
        df_unique[df.index.name] = [freq_list[temp_res][0] + f"_{i:04}" for i in range(1, len(df_unique) + 1)]
        df_unique.set_index(df_unique.columns[-1], inplace=True)

        # Create a mapping from old index to new index
        id_mapping = {}
        # Iterate over each row in the original DataFrame
        for old_index, row in df.iterrows():
            # Find the corresponding new index in the DataFrame containing unique rows
            new_index = df_unique[df_unique.eq(row).all(axis=1)].index[0]
            # Add the mapping to the dictionary
            id_mapping[old_index] = new_index

        self._check_temporal_mapping(df_unique, id_mapping, temp_res)

        return df_unique, id_mapping

    def _check_temporal_mapping(self, df_unique: DataFrame, id_mapping: dict, temp_res: str):
        """
        Validates that each profile in `df_unique` sums to the expected total
        based on the temporal resolution (`temp_res`), accounting for leap years
        and consistency settings.

        Parameters:
            df_unique: pd.DataFrame
                Unique time series profiles (rows) across time steps (columns).
            id_mapping: dict
                Maps original IDs to `df_unique` row indices.
            temp_res: str
                Temporal resolution: "hourly", "daily", "weekly", or "monthly".

        Raises:
            ValueError: If profile sums or column counts are incorrect and consistency is enforced.
        Logs:
            Warnings if inconsistencies are found and `daily_consistency` is False.

        Returns:
            None
        """

        # check profiles: that the number of columns is equal to the sum (because it can still be the correct sum but
        # incorrect column numbers).
        if self.daily_consistency:  # TODO: rename flag to profile/temporal_consistency
            if not allclose(df_unique.sum(axis=1), len(df_unique.columns), atol=0.1):
                # Find the index of the row that does not sum to the expected value in freq_count
                row_index = df_unique[~isclose(df_unique.sum(axis=1), len(df_unique.columns), atol=0.1)].index
                old_index = []
                for i_row_index in row_index:
                    old_index += [k for k, v in id_mapping.items() if v == i_row_index]
                raise ValueError(f"The temporal {temp_res} profile {old_index} \
                                    does not sum to the number of columns {len(df_unique.columns)} of the profiles df.")

        if temp_res == "daily":
            # check the profiles df
            isleap_profiles = self.isleap_profiles_check(len(df_unique.columns))
            self.isleap_profiles = isleap_profiles
            years_in_simulation = sorted(list(set([i.year for i in self.time_step_list])))
        else:
            years_in_simulation = [self.time_step_list[0].year]

        # Check that profiles sum up to the number they are supposed to based on the temporal resolution
        error_msg = []
        for year in years_in_simulation:
            days_in_year = 365 + isleap(year)
            freq_count = {"hourly": 24.0, "daily": days_in_year, "weekly": 7.0, "monthly": 12.0}

            # check that sum is equal to the number we expect
            if not allclose(df_unique.sum(axis=1), freq_count[temp_res], atol=0.1):
                if temp_res == "daily" and not self.daily_consistency:
                    # TODO: why both of these conditions? what happens in else?
                    isleap_beginning = isleap(self.time_step_list[0].year)
                    isleap_end = isleap(self.time_step_list[-1].year)

                    # Raise warnings
                    if isleap_profiles:
                        if not isleap_beginning or not isleap_end:
                            error_msg.append(f"The provided profiles are based on a leap year, with 366 columns. \n \
                                However, at least one year in the current simulation is not a leap year.")
                    else:
                        if isleap_beginning or isleap_end:
                            error_msg.append("The provided profiles are based on a non-leap year, with 365 columns. \n \
                                However, at least one year in the current simulation is a leap year.")

                else:
                    # Find the index of the row that does not sum to the expected value in freq_count
                    row_index = df_unique[~isclose(df_unique.sum(axis=1), freq_count[temp_res], atol=0.1)].index
                    old_index = []
                    for i_row_index in row_index:
                        old_index += [k for k, v in id_mapping.items() if v == i_row_index]
                    error_msg.append(f"For year {year}, the temporal {temp_res} profile {old_index} does not sum "
                                     f"to {freq_count[temp_res]}.")

        if len(error_msg) > 0:
            full_msg = "Some errors detected:\n" + "\n".join(error_msg)
            if self.daily_consistency:
                raise ValueError(full_msg)
            else:
                log_message(full_msg, level=7)

        return None

    def __map_emission_p_temp__(self, emissions: DataFrame, id_mapping: Dict, temp_res: str) -> DataFrame:
        """
        Map the temporal profiles to the emissions DataFrame based on the provided mapping.

        This function processes the temporal profiles for each row in the emissions DataFrame.
        For 'hourly' resolution, it parses the profile string into a dictionary (if it contains multiple profiles)
        and validates that each profile exists in `id_mapping`, replacing it with the mapped value.
        For other resolutions ('daily', 'weekly', 'monthly'),
        it validates and replaces the profile string directly with the mapped value from `id_mapping`.

        Parameters
        ----------
        emissions : DataFrame
            The DataFrame containing the emission data. It should have a column named 'P_hour', 'P_week',
            or 'P_month' depending on the temporal resolution (temp_res).

        id_mapping : Dict
            A dictionary mapping the old profile IDs to the new profile IDs for unique rows.

        temp_res : str
            The resolution of the temporal profile. It can be 'hourly', 'daily', 'weekly', or 'monthly'.

        Returns
        -------
        DataFrame
            The emissions DataFrame with either parsed profile dictionaries (for 'hourly') or mapped strings
            (for other resolutions).

        Raises
        ------
        ValueError
            If there are profile IDs in the emissions DataFrame that are not present in the temporal profiles.
            A single ValueError is raised with a combined message for all missing profiles.

        Steps
        -----
        1. Determine the appropriate column to process based on the temporal resolution (e.g., 'P_hour' for 'hourly').
        2. Iterate over each row of the emissions DataFrame:
           - For 'hourly' resolution:
               a. Parse the profile string (e.g., "weekday=H_weekday, weekend=H_weekend") into a dictionary.
               b. For each entry in the dictionary (e.g., 'weekday', 'weekend'), check if the profile exists in
                  `id_mapping`.
               c. If it exists, replace it with the corresponding mapped value from `id_mapping`.
               d. Store the modified dictionary in the DataFrame.
           - For other resolutions ('daily', 'weekly', 'monthly'):
               a. Validate that the profile string exists in `id_mapping`.
               b. If it exists, replace the profile string with the mapped value.
               c. Store the mapped value in the DataFrame.
        3. If any profiles are missing from `id_mapping`, collect all errors and raise a single `ValueError`
           with a combined message.
        """
        p_freq_name = {"hourly": "P_hour", "daily": "P_day", "weekly": "P_week", "monthly": "P_month"}

        # This is needed because LPS contains duplicated index and this function does not work with them.
        emissions = emissions.reset_index()

        # List to accumulate error messages
        error_messages = []

        # Iterate through each row in the emissions DataFrame
        for idx, row in emissions.iterrows():
            profile_str = row[p_freq_name[temp_res]]

            try:
                if temp_res == 'hourly':
                    # For hourly profiles, parse the profile string into a dictionary
                    profile_mapping = self.__parse_profile_string__(profile_str)

                    # Validate each profile in the dictionary and replace it with the corresponding ID from id_mapping
                    for key, profile_id in profile_mapping.items():
                        if profile_id not in id_mapping:
                            error_messages.append(f"Profile '{profile_id}' not found in id_mapping at row {idx}.")
                        else:
                            profile_mapping[key] = id_mapping[profile_id]  # Replace with mapped value

                    # Store the modified dictionary in the DataFrame
                    emissions.at[idx, p_freq_name[temp_res]] = profile_mapping

                else:
                    # For other resolutions, validate and replace the profile string with its mapped value
                    if profile_str not in id_mapping:
                        error_messages.append(f"Profile '{profile_str}' not found in id_mapping at row {idx}.")
                    else:
                        # Replace the profile string with the mapped value
                        emissions.loc[idx, p_freq_name[temp_res]] = id_mapping[profile_str]
            except ValueError as e:
                raise e
                # error_messages.append(f"Error parsing profile at row {idx}: {e}")

        # If any errors were found, raise a single ValueError with all error messages combined
        if error_messages:
            raise ValueError("\n".join(error_messages))

        # Setting the original index.
        emissions = emissions.set_index(['SNAP_activity', 'NUTS2_code'])

        return emissions

    def __get_final_temp_profile__(self, emission: DataFrame, temp_prof_type: str) -> Tuple[DataFrame, DataFrame]:
        """
        Mapping between P_temp and P_month, P_week, (P_day), P_hour.

        Parameters
        ----------
        emission : DataFrame
            Table containing the emission data.
        temp_prof_type : str
            String indicating which temporal profiles are used. It can be 'month_week' or 'day'.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Tuple containing the emission DataFrame with P_temp and the mapping between
                P_temp and P_month, P_week, P_day, P_hour.
        """

        emission = self.__add_tzid_column__(self.profiles_paths['nuts2_info'], emission)
        if temp_prof_type == 'month_week':
            temp_prof_columns = ['P_month', 'P_week', 'P_hour', 'tzid']
        elif temp_prof_type == 'day':
            temp_prof_columns = ['P_day', 'P_hour', 'tzid']
        else:
            raise ValueError("temp_prof_type should be 'month_week' or 'day'.")

        if self.holidays:
            temp_prof_columns.append('NUTS2_aux')

        # Convert any dict-type entries in P_hour to ordered strings for grouping purposes
        if 'P_hour' in temp_prof_columns:
            emission['P_hour'] = emission['P_hour'].apply(
                lambda x: dict_to_ordered_str(x) if isinstance(x, dict) else x)

        # Perform grouping by temp_prof_columns
        if self.holidays:
            emission['NUTS2_aux'] = emission.index.get_level_values(level='NUTS2_code')

        emission['P_temp'] = emission.groupby(temp_prof_columns).ngroup()
        emission['P_temp'] = emission['P_temp'].apply(lambda temp_id: f"T_{temp_id + 1:04d}")

        # Group by P_temp and keep only the first row, and keep only P_month, P_week, and P_hour columns
        profiles_mapping = emission.reset_index().groupby('P_temp').first().reset_index()
        temp_prof_columns.append('P_temp')
        profiles_mapping = profiles_mapping[temp_prof_columns]

        # Set P_temp as index
        profiles_mapping.set_index('P_temp', inplace=True)

        # Drop unnecessary columns
        emission.drop(columns=['P_month', 'P_week', 'P_day', 'P_hour'], inplace=True)
        if self.holidays:
            profiles_mapping.rename(columns={'NUTS2_aux': 'NUTS2_code'}, inplace=True)
            emission.drop(columns=['NUTS2_aux'], inplace=True)

        return emission, profiles_mapping

    def __unify_seie_lps_mapping__(self, temp_profile_mapping: DataFrame,
                                   temp_profile_mapping_ps: DataFrame,
                                   point_sources):
        """
        Unifies SEIE and point source temporal profile mappings by updating point source profile
        indices to continue the numbering from the SEIE profiles and concatenating both mappings.

        Parameters
        ----------
        temp_profile_mapping : DataFrame
            Table containing SEIE temporal profiles with 'P_temp' as the index.

        temp_profile_mapping_ps : DataFrame
            Table containing point source temporal profiles with 'P_temp' as the index.

        Returns
        -------
        # TODO Fix documentation

        Modifies
        --------
        self.point_sources : DataFrame
            Updates the 'P_temp' column to reflect new indices in the combined mapping.

        Steps
        -----
        1. Modify the indices of `temp_profile_mapping_ps` to continue the numbering of `temp_profile_mapping`,
           ensuring unique indices in the combined mapping.
        2. Create a dictionary mapping the original `P_temp` values in `temp_profile_mapping_ps` to their
           new indices.
        3. Update the 'P_temp' column in `self.point_sources` to point to the newly generated indices in
           `temp_profile_mapping_ps`.
        4. Concatenate `temp_profile_mapping` and `temp_profile_mapping_ps` into a single DataFrame with unified
           temporal profile mappings for both SEIE and point sources.
        """

        # Step 1: Modify indices of temp_profile_mapping_ps to continue numbering after temp_profile_mapping
        last_index = temp_profile_mapping.index[-1]
        last_index_num = int(last_index.split("_")[1])  # Extract numerical part of last index

        # Generate new indices for temp_profile_mapping_ps by incrementing from the last index number
        new_indices = [f"T_{str(last_index_num + i + 1).zfill(4)}" for i in range(len(temp_profile_mapping_ps))]
        original_indices = temp_profile_mapping_ps.index  # Keep original indices for mapping

        # Step 2: Create a mapping dictionary from original P_temp values to new indices
        mapping_dict_ps = dict(zip(original_indices, new_indices))
        temp_profile_mapping_ps = temp_profile_mapping_ps.copy()
        temp_profile_mapping_ps.index = new_indices  # Assign new indices to temp_profile_mapping_ps

        # Step 3: Update the 'P_temp' column in self.point_sources with the correct mapping
        # Here, we map the original P_temp values in self.point_sources to the new indices
        point_sources['P_temp'] = point_sources['P_temp'].map(mapping_dict_ps)

        # Step 4: Concatenate temp_profile_mapping and temp_profile_mapping_ps
        combined_temp_profile_mapping = concat([temp_profile_mapping, temp_profile_mapping_ps])

        return combined_temp_profile_mapping, point_sources

    def _populate_inconsistent_time_step_list(self, utc_times: DataFrame):
        # for each time step get only the year and get a mask with isleap of the years
        time_step_leapyears = [isleap(i) for i in utc_times["year"]]
        # wherever there is a mismatch return True, where they match return False
        if self.isleap_profiles is not None:
            self.inconsistent_time_steps = [i != self.isleap_profiles for i in time_step_leapyears]

    def __calculate_yr_to_hr_profiles__(self, profiles: dict, temp_profile_mapping: DataFrame,
                                        daily: bool = False) -> DataFrame:
        """
        Converts annual profile data into hourly profiles based on monthly, weekly, and hourly scaling factors.
        It adjusts for time zones by converting local hourly profiles to UTC.

        Parameters
        ----------
        profiles : dict of str -> pandas.DataFrame
            Dictionary containing three DataFrames: 'monthly', 'weekly', and 'hourly'.
            Each DataFrame represents the scaling factors for each corresponding period.
            - 'monthly': A DataFrame where the columns are the months (e.g., January, February) and rows represent
                         different profile groups (e.g., 'M_0001', 'M_0002').
            - 'weekly': A DataFrame where the columns are the days of the week (e.g., Monday, Tuesday) and rows
                        represent different profile groups (e.g., 'W_0001', 'W_0002').
            - 'daily': A DataFrame where the columns are the days of the year (0 to 365 or 366) and rows represent
                       different profile groups (e.g., 'D_0001', 'D_0002').
            - 'hourly': A DataFrame where the columns are the hours of the day (0 to 23) and rows represent different
                        profile groups (e.g., 'H_0001', 'H_0002').

        temp_profile_mapping : pandas.DataFrame
            A DataFrame where each row corresponds to a profile template (e.g., 'T_0001'), and columns indicate the
            corresponding 'P_month', 'P_week', 'P_hour', and the 'tzid' (time zone identifier). The columns are:
            - 'P_month': Refers to the monthly profile (e.g., 'M_0020').
            - 'P_week': Refers to the weekly profile (e.g., 'W_0002').
            - 'P_day': Refers to the daily profile (e.g., 'D_0002').
            - 'P_hour': Refers to the hourly profile (e.g., 'H_0002', or a dict with day-specific keys like
                        {'mon': 'H_0001', 'tue': 'H_0002'}).
            - 'tzid': Time zone identifier (e.g., 'Europe/Madrid') for adjusting UTC times to local times.

        daily: bool
            A flag that True when we are converting the profile data for activities which have daily profiles,
            and False when the activities have monthly/weekly profiles. Is False by default.

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row corresponds to an hourly value for the entire year.
            The index will be the `P_temp` values (e.g., 'T_0001', 'T_0002', etc.),
            and the columns will represent each hour (from 0 to 23 for each day of the year).
        """
        # Normalize input profiles (monthly, weekly, hourly)
        if daily:
            profiles["daily"] = profiles["daily"].div(profiles["daily"].sum(axis=1), axis=0)
        else:
            profiles['monthly'] = profiles['monthly'].div(profiles['monthly'].sum(axis=1), axis=0)

        profiles['hourly'] = profiles['hourly'].div(profiles['hourly'].sum(axis=1), axis=0)

        # Create a DataFrame of all utc_times values
        utc_times = DataFrame(self.time_step_list, columns=['utc_time'])
        # Ensure utc_time is localized to UTC first
        utc_times['utc_time'] = to_datetime(utc_times['utc_time'], utc=True)
        # Create empty DataFrame to hold final results
        final_profile = DataFrame(index=temp_profile_mapping.index, columns=range(len(utc_times)))

        # Step 2: Convert UTC to local time for each profile's timezone
        # Use vectorized operations to convert UTC to local time for all profiles at once
        for profile_id, profile_info in temp_profile_mapping.iterrows():

            # Convert from UTC to the respective local timezone
            tzid = profile_info['tzid']
            utc_times['local_time'] = utc_times['utc_time'].apply(lambda x: x.astimezone(timezone(tzid)))
            utc_times = self._add_time_components(utc_times=utc_times, profile_info=profile_info, daily=daily)

            self._populate_inconsistent_time_step_list(utc_times)

            if daily and any(self.inconsistent_time_steps):
                # self._populate_inconsistent_time_step_list(utc_times)
                utc_times, factors = self._extract_temp_factors_leapyears(
                    profile_id, profile_info, profiles, utc_times)
            else:
                utc_times, factors = self._extract_temp_factors(
                    profile_id, profile_info, profiles, utc_times, daily=daily)

            if daily:
                daily_factors, hourly_factors = factors
                # Combine the daily and hourly factors into a single factor for each hour
                combined_factors = daily_factors * hourly_factors / 3600

            else:
                monthly_factors, weekly_factors, hourly_factors = factors
                # Rebalanced weekly factors for each local time month and year (all at once)
                rebalanced_weekly_factors = weekly_factors.copy()
                unique_year_month_pairs = utc_times[['year', 'month']].drop_duplicates().values
                rebalanced_weekly_factors = self._rebalance_weekly(
                    unique_year_month_pairs, utc_times, rebalanced_weekly_factors, profiles, profile_info)
                # Combine the monthly, rebalanced weekly, and hourly factors into a single factor for each hour
                combined_factors = monthly_factors * rebalanced_weekly_factors * hourly_factors / 3600

            # Assign the combined factors to the final profile DataFrame
            final_profile.loc[profile_id] = combined_factors

        return final_profile

    def _add_time_components(self, utc_times, profile_info, daily: bool):
        """
        Adds time components (month, weekday, hour, year) to a DataFrame based on local time.

        If holidays are enabled, weekdays falling on holidays are set to 6 (Sunday) based on
        the region code in `profile_info['NUTS2_code']`.

        Parameters:
        ----------
        utc_times : pandas.DataFrame
            DataFrame with a 'local_time' datetime column.
        profile_info : dict
            Contains metadata, including 'NUTS2_code' for holiday lookup.
        daily: bool
            A flag that True when we are converting the profile data for activities which have daily profiles,
            and False when the activities have monthly/weekly profiles. Is False by default.

        Returns:
        -------
        pandas.DataFrame
            Modified DataFrame with added 'month', 'weekday', 'day', 'hour', and 'year' columns.
        """

        # Extract the components (month, weekday, hour, year) of the local time for future use
        utc_times['month'] = utc_times['local_time'].dt.month
        utc_times['month_name'] = utc_times['local_time'].dt.month_name()
        utc_times['weekday'] = utc_times['local_time'].dt.weekday
        utc_times['weekday_name'] = utc_times['local_time'].dt.day_name()

        if self.holidays:
            # Create a column indicating if it's a holiday
            utc_times['is_holiday'] = utc_times['local_time'].apply(lambda x: self.__is_holiday__(
                x, profile_info['NUTS2_code']))
            # Modify 'weekday' to 6 (Sunday) when 'is_holiday' is True
            utc_times.loc[utc_times['is_holiday'], 'weekday'] = 6
            # Remove the 'is_holiday' column as it's no longer needed
            utc_times.drop(columns=['is_holiday'], inplace=True)
        else:
            # No action
            pass

        utc_times['hour'] = utc_times['local_time'].dt.hour
        utc_times['year'] = utc_times['local_time'].dt.year

        if daily:
            utc_times["day"] = utc_times["local_time"].dt.day
            utc_times["day_of_year"] = to_datetime({"year": utc_times["year"],
                                                    "month": utc_times["month"],
                                                    "day": utc_times["day"]}).dt.dayofyear
        return utc_times

    def _extract_temp_factors_leapyears(self, profile_id: str, profile_info: dict, profiles: dict,
                                        utc_times: DataFrame) -> list:
        """
        Rebalance daily profile factors for leap year or non-leap year scenarios.

        This function adjusts the daily profile factors based on whether the
        current year is a leap year or not. It ensures consistency by generating
        the appropriate complementary daily profile (with or without day 366),
        normalizing it, and replacing values for time steps marked as inconsistent
        with the original profile.

        Parameters
        ----------
        utc_times : pandas.DataFrame
            A DataFrame containing time step information. Must include a "day"
            column with values ranging from 1 to 365 or 366, depending on the year.

        daily_factors : list or array-like
            A list of scaling factors applied to each day based on the selected
            profile. These are adjusted where inconsistencies are identified.

        profiles : dict
            A dictionary containing different types of profiles. Must include a
            "daily" key whose value is a DataFrame of daily profiles with days
            ("1" to "365", and optionally "366") as columns.

        profile_info : dict
            Metadata for the specific profile in use. Must include the key "P_day"
            identifying which row to use from the `profiles["daily"]` DataFrame.

        Returns
        -------
        list
            A list of daily factors where inconsistent time steps have been
            replaced with corresponding values from the adjusted daily profile
            (either leap year or non-leap year, depending on context).
        """
        hour_profile = str_to_dict(profile_info['P_hour'])
        utc_times['profile'] = utc_times['weekday'].apply(lambda day: self.__select_profile__(day, hour_profile))
        if utc_times['profile'].isnull().any():
            raise ValueError(f"Missing profile in hour_profile for P_temp {profile_id}. "
                             f"Ensure there is a valid key in the dictionary.")
        hourly_factors = utc_times.apply(
            lambda row: profiles['hourly'].loc[row['profile']].values[row['hour']], axis=1).values

        daily_profile_id = profile_info["P_day"]
        daily_profile = profiles["daily"].loc[[daily_profile_id]]

        # create another profile for the missing scenario (leap or nonleap year)
        new_daily_profile = daily_profile.copy()
        if self.isleap_profiles:
            new_daily_profile = new_daily_profile.drop(columns=["366"], axis=1)
        else:
            new_daily_profile["366"] = new_daily_profile["365"]

        # re-normalize the other daily profile
        new_daily_profile = new_daily_profile.div(new_daily_profile.sum(axis=1), axis=0)

        # extract day-of-year column names as strings
        day_of_year_columns = [str(i) for i in utc_times["day_of_year"]]

        # retrieve the values from the profiles. use the profile that corresponds to the first time step
        if self.inconsistent_time_steps[0]:
            # if the first time step is inconsistent, use the new daily profile
            new_daily_factors = list(new_daily_profile.loc[daily_profile_id, day_of_year_columns])
        else:
            # if the first time step is consistent, use the original daily profile
            new_daily_factors = list(daily_profile.loc[daily_profile_id, day_of_year_columns])

        factors = (new_daily_factors, hourly_factors)

        return utc_times, factors

    def _extract_temp_factors(self, profile_id, profile_info, profiles, utc_times, daily: bool):
        """
        Extracts the monthly, weekly, and hourly temporal factors for a given profile.

        Based on the `profile_info`, this method:
        - Determines the appropriate hourly sub-profile for each timestamp using the weekday.
        - Retrieves corresponding monthly, weekly, and hourly scaling factors from the `profiles` data.

        Parameters:
        ----------
        profile_id : str
            Identifier for the temporal profile (used for error messages).

        profile_info : dict
            Dictionary containing profile metadata, including:
            - 'P_day': daily profile key
            - 'P_month': monthly profile key
            - 'P_week' : weekly profile key
            - 'P_hour' : JSON string representing weekday-to-hourly-profile mapping

        profiles : dict
            Dictionary with 'monthly', 'daily', 'weekly', and 'hourly' DataFrames, indexed by profile ID.

        utc_times : pandas.DataFrame
            DataFrame with columns: 'month', 'day', 'weekday', and 'hour' (all integers).
            Modified in-place to include a 'profile' column for hourly profile mapping.

        Returns:
        -------
        tuple of np.ndarray
            A tuple containing:
            - utc_times
            - tuple of:
                - monthly_factors: Array of monthly scaling values
                - weekly_factors : Array of weekly scaling values
                - hourly_factors : Array of hourly scaling values

            or if daily = True
            A tuple containing:
            - utc_times
            - tuple of:
                - daily_factors: Array of daily scaling values
                - hourly_factors: Array of hourly scaling values

        Raises:
        -------
        ValueError
            If any weekday is missing from the hourly profile mapping.
        """
        # Apply the function to select the profile for each hour based on the local time weekday
        hour_profile = str_to_dict(profile_info['P_hour'])
        utc_times['profile'] = utc_times['weekday'].apply(lambda day: self.__select_profile__(day, hour_profile))
        if utc_times['profile'].isnull().any():
            raise ValueError(f"Missing profile in hour_profile for P_temp {profile_id}. "
                             f"Ensure there is a valid key in the dictionary.")
        hourly_factors = utc_times.apply(
            lambda row: profiles['hourly'].loc[row['profile']].values[row['hour']], axis=1).values

        if daily:
            daily_profile = profile_info["P_day"]
            day_of_year_columns = [str(i) for i in utc_times["day_of_year"]]
            daily_factors = list(profiles["daily"].loc[daily_profile, day_of_year_columns])
            factors = (daily_factors, hourly_factors)

        else:
            month_profile = profile_info['P_month']
            week_profile = profile_info['P_week']

            monthly_factors = list(profiles["monthly"].loc[month_profile, utc_times["month_name"]])
            weekly_factors = list(profiles["weekly"].loc[week_profile, utc_times["weekday_name"]])

            factors = (monthly_factors, weekly_factors, hourly_factors)

        return utc_times, factors

    def _rebalance_weekly(self, unique_year_month_pairs, utc_times, rebalanced_weekly_factors,
                          profiles, profile_info):
        """
        Rebalances weekly profile values for each unique (year, month) pair based on the actual weekdays
        present in the `utc_times` DataFrame. If holidays are enabled, it uses a holiday-aware rebalancing method that
        adjusts for holidays based on the regional code in `profile_info['NUTS2_code']`.

        Parameters:
        ----------
        unique_year_month_pairs : list of tuple
            List of (year, month) pairs present in the dataset.
        utc_times : pandas.DataFrame
            DataFrame containing 'year', 'month', and 'weekday' columns for each timestamp.
        rebalanced_weekly_factors : numpy.ndarray
            Preallocated array to store rebalanced weekly factors for each timestamp.
        profiles : dict
            Dictionary of temporal profiles. Must include a 'weekly' DataFrame.
        week_profile : str or int
            Key/index to select the appropriate weekly profile from `profiles['weekly']`.
        profile_info : dict
            Metadata dictionary that includes 'NUTS2_code' for region-specific holiday logic.

        Returns:
        -------
        numpy.ndarray
            Updated `rebalanced_weekly_factors` with values adjusted for each months weekday distribution.
        """
        week_profile = profile_info['P_week']

        # Iterate over each unique year and month in the local time
        for year, month in unique_year_month_pairs:
            mask = (utc_times['year'] == year) & (utc_times['month'] == month)
            days_in_month = utc_times.loc[mask, 'weekday']

            # Rebalance the weekly profile for the exact days within the month
            rebalanced_factors = self.__rebalance_weekly_profile__(
                weekly_profile=profiles["weekly"].loc[week_profile], year=year, month=month,
                profile_info=profile_info, holidays=self.holidays, holiday_regions=self.holiday_regions,
            ).values[days_in_month]

            rebalanced_weekly_factors = array(rebalanced_weekly_factors)
            rebalanced_factors = array(rebalanced_factors)
            rebalanced_weekly_factors[mask] = rebalanced_factors

        return rebalanced_weekly_factors

    @staticmethod
    def __rebalance_weekly_profile__(weekly_profile: Series, year: int, month: int, profile_info,
                                     holidays: bool, holiday_regions=None):

        # Get the number of days in the specified month
        num_days_in_month = monthrange(year, month)[1]
        # Create a range of all days in that month
        days_in_month = date_range(start=f"{year}-{month}-01", periods=num_days_in_month)
        # Count how many times each weekday occurs in the month (Monday=0, ..., Sunday=6)

        if holidays:
            nuts2_code = profile_info['NUTS2_code']
            # Get the list of holidays for the region and country
            country_code = nuts2_code[:2].upper()
            national_holidays = country_holidays(country_code, years=[year])
            regional_holidays = country_holidays(country_code, subdiv=holiday_regions[nuts2_code], years=[year])

            # Combine national and regional holidays for the month
            holidays = [day for day in days_in_month if day in national_holidays or day in regional_holidays]

            # Count occurrences of each weekday, treating holidays as Sundays (6)
            weekday_counts = days_in_month.to_series().apply(
                lambda day: 6 if day in holidays else day.weekday()
            ).value_counts().sort_index()

        else:
            weekday_counts = days_in_month.to_series().dt.dayofweek.value_counts().sort_index()

        # Rename the weekly_profile to have integer indices from 0 (Monday) to 6 (Sunday)
        weekly_profile = weekly_profile.rename(index={
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        })

        # Compute the total sum of the weighted profile (this will be rebalanced)
        profile_sum = (weekly_profile * weekday_counts).sum()

        # Scale the profile to ensure that the sum of the weighted profile matches the total number of days in the month
        adjustment_factor = weekday_counts.sum() / profile_sum
        rebalanced_profile = weekly_profile * adjustment_factor

        # Normalize the profile so that (rebalanced_profile * weekday_counts).sum() equals 1
        normalization_factor = (rebalanced_profile * weekday_counts).sum()  # number of days
        normalized_profile = rebalanced_profile / normalization_factor

        return normalized_profile

    @staticmethod
    def __add_tzid_column__(map_nut_tzid_path: str, emissions: DataFrame) -> DataFrame:
        """
        Adds a 'tzid' column to the emissions DataFrame based on NUTS2 codes and a mapping from a YAML file.

        Parameters
        ----------
        map_nut_tzid_path : str
            Path to the file that contains the mapping between NUTS2 codes and timezone information.

        emissions : DataFrame
            A pandas DataFrame containing emissions data. The index of this DataFrame must
            have a level named `'NUTS2_code'` which corresponds to the NUTS2 regional code
            to be used for mapping timezone identifiers.

        Returns
        -------
        DataFrame
            A modified emissions DataFrame with a new column `'tzid'`, containing the timezone
            identifier corresponding to each NUTS2 region.

        Raises
        ------
        KeyError
            If a NUTS2 code in the emissions DataFrame does not have a corresponding timezone in the YAML file.

        FileNotFoundError
            If the file specified by `'map_nut_tzid_path'` does not exist.
        """
        # Load the YAML file
        with open(map_nut_tzid_path, 'r') as file:
            nuts2_tzid_data = load_yaml(file)

        # Create a dictionary mapping NUTS2 codes to their respective timezones
        nuts2_tzid_dict = {code: data['Timezone'] for code, data in nuts2_tzid_data.items()}

        # Map the 'NUTS2_code' level in the index to the timezone using the dictionary
        nuts2_codes = emissions.index.get_level_values('NUTS2_code')
        emissions['tzid'] = nuts2_codes.map(nuts2_tzid_dict)

        # Check for any missing mappings and raise an error if found
        if emissions['tzid'].isnull().any():
            missing_codes = nuts2_codes[emissions['tzid'].isnull()].unique()
            raise KeyError(f"Missing timezone information for NUTS2 codes: {missing_codes}")

        return emissions

    @staticmethod
    def __load_nuts2_holidays_region__(map_nut_tzid_path: str) -> dict:
        """
        Loads a YAML file mapping NUTS2 codes to their corresponding Holidays_Region codes and returns a dictionary.

        Parameters
        ----------
        map_nut_tzid_path : str
            Path to the YAML file that contains the mapping between NUTS2 codes and Holidays_Region information.

        Returns
        -------
        dict
            A dictionary where keys are NUTS2 codes and values are the corresponding Holidays_Region codes.
            Only NUTS2 codes with a defined Holidays_Region are included in the dictionary.

        Raises
        ------
        FileNotFoundError
            If the file specified by 'map_nut_tzid_path' does not exist.
        """
        # Load the YAML file
        with open(map_nut_tzid_path, 'r') as file:
            nuts2_data = load_yaml(file)

        # Extract only entries with Holidays_Region defined
        holidays_region_dict = {
            code: data['Holidays_Region'] for code, data in nuts2_data.items() if 'Holidays_Region' in data
        }

        return holidays_region_dict

    def __is_holiday__(self, local_date, nuts2_code):
        """
        Checks if a given date is a holiday, either national or regional, based on the NUTS2 code.

        Parameters
        ----------
        local_date : datetime
            The local date to check.
        nuts2_code : str
            The NUTS2 code representing the regional area to use for determining holidays.

        Returns
        -------
        bool
            True if the date is a holiday (either national or regional), False otherwise.
        """
        # Get national and regional holidays
        national_holidays = country_holidays(nuts2_code[:2].upper())
        regional_holidays = country_holidays(country=nuts2_code[:2].upper(), subdiv=self.holiday_regions[nuts2_code])

        return local_date.date() in national_holidays or local_date.date() in regional_holidays

    @staticmethod
    # Define a function to select the correct profile based on weekday
    def __select_profile__(weekday, profile_dict):
        if weekday == 0:  # Monday
            return profile_dict.get('mon') or profile_dict.get('weekday') or profile_dict.get('default')
        elif weekday == 1:  # Tuesday
            return profile_dict.get('tue') or profile_dict.get('weekday') or profile_dict.get('default')
        elif weekday == 2:  # Wednesday
            return profile_dict.get('wed') or profile_dict.get('weekday') or profile_dict.get('default')
        elif weekday == 3:  # Thursday
            return profile_dict.get('thu') or profile_dict.get('weekday') or profile_dict.get('default')
        elif weekday == 4:  # Friday
            return profile_dict.get('fri') or profile_dict.get('weekday') or profile_dict.get('default')
        elif weekday == 5:  # Saturday
            return profile_dict.get('sat') or profile_dict.get('weekend') or profile_dict.get('default')
        elif weekday == 6:  # Sunday
            return profile_dict.get('sun') or profile_dict.get('weekend') or profile_dict.get('default')
        else:
            raise ValueError(f"Invalid weekday {weekday}")


def calculate_date_array(start_date: datetime, timestep_type: str, timestep_num: int,
                         timestep_freq: int) -> List[datetime]:
    """
    Generates an array of date-times based on the provided timestep type, number, and frequency.

    This function calculates the series of date-times starting from `start_date` based on the
    timestep type ('hourly', 'daily', 'monthly', or 'yearly') and the frequency specified by
    `timestep_freq`. The number of timesteps is controlled by `timestep_num`.

    Parameters
    ----------
    start_date : datetime
        The date and time of the first timestep.

    timestep_type : str
        The type of timestep. It can be 'hourly', 'daily', 'monthly', or 'yearly'.

    timestep_num : int
        The number of timesteps to generate. Must be greater than zero.

    timestep_freq : int
        The frequency of timesteps. For example, if `timestep_type` is 'hourly' and `timestep_freq` is 2,
        the interval between each timestep will be 2 hours.

    Returns
    -------
    List[datetime]
        A list of date-times to simulate based on the provided parameters.

    Raises
    ------
    ValueError
        If `timestep_num` is zero or `timestep_freq` is zero or an invalid `timestep_type` is provided.

    Notes
    -----
    - The function raises a `ValueError` if `timestep_num` is equal to 0, as at least one timestep is required.
    - The `relativedelta` function from the `dateutil` library is used to handle month and year increments.
    """
    if timestep_num == 0:
        raise ValueError("timestep_num must be greater than 0, but received 0")
    if timestep_freq == 0:
        raise ValueError("timestep_freq must be greater than 0, but received 0")

    date_array = [start_date]

    # Helper to calculate the next date
    def get_next_date(current_date: datetime, tstep_type: str, tstep_freq: int) -> datetime:
        if tstep_type == 'hourly':
            return current_date + timedelta(hours=tstep_freq)
        elif tstep_type == 'daily':
            return current_date + timedelta(days=tstep_freq)
        elif tstep_type == 'monthly':
            return current_date + relativedelta(months=tstep_freq)
        elif tstep_type == 'yearly':
            return current_date + relativedelta(years=tstep_freq)
        else:
            raise ValueError(f"Invalid timestep_type: {tstep_type}")

    # Loop to generate date array
    for _ in range(1, timestep_num):
        next_date = get_next_date(date_array[-1], timestep_type, timestep_freq)
        date_array.append(next_date)

    return date_array
