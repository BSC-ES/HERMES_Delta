from typing import Dict, Tuple
from pandas import DataFrame, concat
from numpy import zeros, nonzero, float64, array, where, isin
from yaml import safe_load as load_yaml
from hermes_d.config import log_message


class VerticalDelta(object):
    """
    Class to handle vertical distribution of emissions in the HERMES model.

    This class is responsible for reading vertical profiles from a YAML file, mapping them to sector information,
    and distributing emissions vertically based on the provided profiles. It also calculates vertical factors
    for each output layer and distributes emissions accordingly.

    Parameters
    ----------
    profiles_paths : str
        The path to the YAML file containing vertical profiles.
    sector_info : DataFrame
        A DataFrame containing sector information with 'P_vert' column representing vertical profiles.
    output_layers : list
        A list of vertical layers or levels to which emissions will be distributed.
    """

    def __init__(self, profiles_paths: str, sector_info: DataFrame, output_layers: list):
        self.profiles_paths = profiles_paths
        self.sector_info = sector_info
        self.output_layers = output_layers

    def initialize(self):
        """
        Initialize the VerticalDelta class by obtaining vertical profiles and calculating vertical factors.

        This method reads vertical profiles from the specified YAML file, maps them to the sector information,
        and calculates the vertical factors for each output layer based on the obtained profiles.

        Returns
        -------
        Tuple[DataFrame, Dict]
            First item is a DataFrame with the updated sector information with the 'P_vert' column mapped to unique
            profile categories.
            Second item is a dictionary with vertical profile ID as keys and numpy array as values.
            The numpy array contains the calculated vertical factors for each output layer.
        """
        vertical_profiles, mapping = self.get_vertical_profiles()
        vertical_factors = self.calculate_vertical_factors(vertical_profiles)
        self.sector_info = self.map_sector_info_p_vert(mapping)
        return self.sector_info, vertical_factors

    # noinspection DuplicatedCode
    def get_vertical_profiles(self) -> Tuple[DataFrame, Dict, Dict]:
        """
        Obtain the vertical profile for emissions and organize them into unique profiles.

        This function reads vertical profiles from a YAML file and maps them to the sector information
        based on the 'P_vert' column.

        Returns
        -------
        Tuple[Dict, Dict]
            First item is a dictionary containing duplicate vertical profiles.
            Second item is a dictionary mapping original vertical profiles to unique categories (V_xxxx).

        Raises
        ------
        KeyError
            If any 'P_vert' in sector_info is not listed in the vertical profiles.
        ValueError
            If any vertical profile has a height higher than the maximal output layer or
            if the values in the profile do not sum up to 1.

        Notes
        -----
        - If 'only_horizontal' is True, all emissions are assumed to be at the surface with a profile of 100%
          at level 0.
        - The function checks the validity of vertical profiles, ensures they sum up to 1, and maps them to unique
        categories.

        """
        vertical_profiles_path = self.profiles_paths["Vertical_profiles"]
        # Read the yaml file
        with open(vertical_profiles_path, 'r') as file:
            v_profile = load_yaml(file)["Vertical_profiles"]

        p_vert = self.sector_info["P_vert"]

        if p_vert.isna().any():
            p_vert_na = p_vert.isna()
            self.sector_info.loc[p_vert_na, "P_vert"] = "surface"
            p_vert = self.sector_info["P_vert"]
            v_profile["surface"] = {0: 1.0}
            ids = self.sector_info.index.to_frame()[p_vert_na].apply("_".join, axis=1)
            warn_msg = "WARNING: SEIE data " + " ".join(ids) + " 'P_vert' is NA. Assign it as surface."
            log_message(warn_msg, level=7)

        # Check: if P_vert are all in v_profile
        if not all(item in v_profile.keys() for item in p_vert):
            tmp = p_vert[~p_vert.isin(v_profile.keys())]
            err_msg = "P_vert " + ' '.join(tmp) + " not listed in the vertical profiles."
            raise KeyError(err_msg)

        v_profile = {key: v_profile[key] for key in p_vert if key in v_profile}
        duplicates = {}
        mapping = {}
        category_counter = 1

        for key, value in v_profile.items():
            # Check: If the heights are all below output maximal layer
            if any(key_aux > self.output_layers[-1] for key_aux in value.keys()):
                msg = "Vertical profile " + key + " has height higher than maximal output layer."
                raise ValueError(msg)

            # Check: If the values sum up to 1
            tolerance = 0.0001
            if abs(sum(value.values()) - 1) > tolerance:
                msg = "Vertical profile " + key + " doesn't sum up to 1."
                raise ValueError(msg)
            category_found = False

            # Check if the value already exists in duplicates
            for category, existing_value in duplicates.items():
                if value == existing_value:
                    mapping[key] = category
                    category_found = True
                    break

            # If value is not found in duplicates, create a new category
            if not category_found:
                category_key = f'V_{str(category_counter).zfill(4)}'
                duplicates[category_key] = value
                mapping[key] = category_key
                category_counter += 1

        return duplicates, mapping

    def map_sector_info_p_vert(self, mapping: dict) -> DataFrame:
        """
        Map the 'P_vert' column in sector_info with new vertical profile keys.

        This function maps the 'P_vert' column values in the sector_info DataFrame to new keys
        based on the provided mapping dictionary.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping original vertical profile keys to new keys.

        Returns
        -------
        DataFrame
            A DataFrame containing the sector information with the 'P_vert' column updated
            with the new vertical profile keys.
        """

        self.sector_info['P_vert'] = self.sector_info['P_vert'].map(mapping)

        return self.sector_info

    def calculate_vertical_factors(self, vertical_profiles: dict):
        """
        Calculate the vertical factors for each vertical output layer based on given vertical profiles.

        This function calculates the vertical factors for each output layer based on the provided
        vertical profiles. The vertical factors represent the distribution weights for each output layer.

        Parameters
        ----------
        vertical_profiles : dict
            A dictionary containing vertical profile IDs as keys and their respective weights
            for different vertical layers as values.

        Returns
        -------
        Dict
            A dictionary with vertical profile ID as keys and numpy array as values.
            The numpy array contains the calculated vertical factors for each output layer.
        """
        result_dict = {}
        for vp_id, vp in vertical_profiles.items():

            vert_fact = zeros(len(self.output_layers))
            prev_layer = 0
            for layer, weight in vp.items():
                if weight != float(0):
                    for element in self.get_weights(prev_layer, layer, weight, self.output_layers):
                        vert_fact[element['index']] += element['weight']
                prev_layer = layer
            vert_fact = array(vert_fact, dtype=float64)

            # Normalizing
            vert_fact = vert_fact / vert_fact.sum()

            result_dict[vp_id] = vert_fact

        return result_dict

    # noinspection DuplicatedCode
    @staticmethod
    def get_weights(prev_layer, layer, in_weight, output_vertical_profile):
        """
        Calculate the weights for distributing a given weight across vertical layers.

        This function calculates the weights for distributing a given weight across the vertical layers
        between a previous layer and a current layer. It provides the weights for each layer within
        the specified altitude range.

        Parameters
        ----------
        prev_layer : float
            Altitude of the lower layer. 0 if it's the first layer.
        layer : float
            Altitude of the current layer.
        in_weight : float
            Input weight to be distributed across the layers.
        output_vertical_profile : list
            A list of vertical layers or altitudes to which emissions will be distributed.

        Returns
        -------
        list
            A list of dictionaries containing weights for each layer.
            Each dictionary has keys 'index' indicating the index of the layer in the output_vertical_profile
            and 'weight' indicating the weight to be distributed to that layer.
        """

        output_vertical_profile_aux = [s for s in output_vertical_profile if s >= prev_layer]
        output_vertical_profile_aux = [s for s in output_vertical_profile_aux if s < layer]
        output_vertical_profile_aux = [prev_layer] + output_vertical_profile_aux + [layer]

        index = len([s for s in output_vertical_profile if s < prev_layer])
        if layer == prev_layer:  # layer = 0, all the weight is at surface
            weight_list = [{'index': 0, 'weight': in_weight}]
        else:
            origin_diff_factor = in_weight / (layer - prev_layer)
            weight_list = []
            for i in range(len(output_vertical_profile_aux) - 1):
                weight = (abs(output_vertical_profile_aux[i] - output_vertical_profile_aux[i + 1])) * origin_diff_factor
                weight_list.append({'index': index, 'weight': weight})
                index += 1

        return weight_list


def distribute_seie_vertically(vertical_factors: Dict, emissions: DataFrame, output_layers: list,
                               pollutants: list) -> DataFrame:
    """
    Distribute the SEIE emissions vertically by adding the corresponding levels into each emission.

    This function takes vertical distribution factors for each level and distributes emissions
    vertically by adding the corresponding levels into each emission based on the provided factors.
    The emissions are adjusted by multiplying them with the corresponding factors.

    Parameters
    ----------
    vertical_factors : Dict
        A dictionary containing the weight for each level. The keys represent different profiles,
        and the values are lists of factors corresponding to each output layer.
    emissions : DataFrame
        A DataFrame containing sector information with emissions data.
        The DataFrame must have a 'P_vert' column to identify profiles.
    output_layers : list
        A list of output layers or levels for which vertical distribution should be performed.
    pollutants : list
        A list of pollutants to which the vertical distribution factors will be applied.

    Returns
    -------
    DataFrame
        Returns a DataFrame with the same structure as the input DataFrame, but with the original pollutants replaced
        with chemical mechanism species. The "P_vert" column is also removed.
    """

    emissions["level"] = 0
    for prof, factor in vertical_factors.items():
        nonzero_len = len(nonzero(factor)[0])
        row_to_replicate = emissions.loc[emissions["P_vert"] == prof].reset_index()
        repeated_row = row_to_replicate.loc[row_to_replicate.index.repeat(nonzero_len)]

        # Fill the level column and multiply emissions by factor
        count = 0
        for ind in range(len(output_layers)):
            if factor[ind] != 0:
                row_inds = [i for i in range(count, repeated_row.shape[0] - nonzero_len + count + 1, nonzero_len)]
                repeated_row.iloc[row_inds, where(repeated_row.columns == "level")[0]] = ind
                repeated_row.iloc[row_inds, where(isin(repeated_row.columns, pollutants))[0]] *= factor[ind]
                count += 1

        repeated_row = repeated_row.set_index(["SNAP_activity", "NUTS2_code"])
        emissions = concat([emissions.loc[emissions["P_vert"] != prof], repeated_row])

    # Remove P_vert column
    emissions = emissions.drop(columns='P_vert')

    return emissions


def distribute_point_sources_vertically(emissions: DataFrame, output_layers: list) -> DataFrame:
    """
    Distribute the emissions vertically by adding the corresponding level into each emission.

    This function distributes emissions vertically by adding the corresponding level into each emission
    based on the height and plume rise factor. It assigns a 'level' to each emission based on the calculated
    height and checks if any calculated height exceeds the top output layer.

    Parameters
    ----------
    emissions : DataFrame
        A DataFrame containing point source information with emission details.
        It should have 'height' and 'plume_rise_factor' columns representing the height of emissions
        and the plume rise factor, respectively.
    output_layers : list
        A list of vertical layers or levels to which emissions will be distributed.

    Returns
    -------
    DataFrame
        A DataFrame containing the point source information with the 'level' column added
        to indicate the vertical level of emissions. The 'height' and 'plume_rise_factor'
        columns are removed from the DataFrame.
    """

    emissions['level'] = emissions['height'] * emissions['plume_rise_factor']
    # Check: if any calculated height is over the top level
    if (emissions['level'] > output_layers[-1]).any():
        tmp = emissions['level'] > output_layers[-1]
        err_msg = "Code " + ' '.join(emissions.loc[tmp, "Code"]) + \
                  " calculated height is higher than the top output layer " + str(output_layers[-1]) + "."
        raise ValueError(err_msg)

    # Mapping function to assign levels based on output_layers
    def map_level(level):
        if level < output_layers[0]:
            return 0
        else:
            for i in range(len(output_layers) - 1):
                if output_layers[i] <= level < output_layers[i + 1]:
                    return i + 1
        return None

    # Map levels (0, 1, etc.) to the 'level' column
    emissions['level'] = emissions['level'].map(map_level)

    # Drop height and plume_rise_factor columns
    emissions = emissions.drop(columns=['height', 'plume_rise_factor'])

    return emissions
