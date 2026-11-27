from .parsing import dict_to_ordered_str, str_to_dict
from .geographical import get_line_lengths, get_polygon_areas
from .geographical import get_line_length_fractions, get_polygon_area_fractions
from .geographical import get_raster_polygon_areas, get_raster_area_fractions
from .parsing import (parse_list, parse_path, parse_end_date, parse_float_list, parse_start_date, parse_bool,
                      parse_vertical_description)

__all__ = ["redistribute_dataframes", "redistribute_dataframes_with_balancing",
           "get_line_lengths", "get_polygon_areas",
           "get_line_length_fractions", "get_polygon_area_fractions",
           "get_raster_polygon_areas", "get_raster_area_fractions",
           "parse_list", "parse_path", "parse_end_date", "parse_float_list", "parse_start_date", "parse_bool",
           "parse_vertical_description",
           "dict_to_ordered_str", "str_to_dict",
           "sum_result", "sum_emissions"]


def __getattr__(name):
    if name in ["redistribute_dataframes", "redistribute_dataframes_with_balancing"]:
        from .redistribute_df import redistribute_dataframes, redistribute_dataframes_with_balancing
        return {
            "redistribute_dataframes": redistribute_dataframes,
            "redistribute_dataframes_with_balancing": redistribute_dataframes_with_balancing,
        }[name]
    if name in ["sum_result", "sum_emissions"]:
        from .sandbox import sum_result, sum_emissions
        return {"sum_result": sum_result, "sum_emissions": sum_emissions}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
