from ast import literal_eval
from datetime import datetime
from hermes_d.config.constants import precision
from numpy import array
from configargparse import ArgumentTypeError


def dict_to_ordered_str(dictionary: dict) -> str:
    """
    Convert a dictionary to a string with keys sorted, ensuring that
    dictionaries with the same content but different key order produce the same string.

    Parameters
    ----------
    dictionary : dict
        The dictionary to convert to a string.

    Returns
    -------
    str
        A string representation of the dictionary with sorted keys.
    """
    return str(sorted(dictionary.items()))


def str_to_dict(string_rep: str) -> dict:
    """
    Convert a string representation of a dictionary back to a dictionary.

    Parameters
    ----------
    string_rep : str
        The string representation of the dictionary.

    Returns
    -------
    dict
        The dictionary converted from the string.

    Notes
    -----
    Assumes that the string was generated using dict_to_ordered_str().
    """
    return dict(literal_eval(string_rep))


def parse_end_date(end_date, start_date):
    """
    Parse the end date.
    If it's not defined it will be the same date that start_date (to do only one day).

    Parameters
    ----------
    end_date : str or datetime
        Date of the last day to simulate.
    start_date : datetime
        Date of the first day to simulate.

    Returns
    -------
    datetime
        Date to the last day to simulate in datetime format.
    """
    if end_date is None:
        return start_date
    else:
        return parse_start_date(end_date)


def parse_list(string: str) -> list:
    """
    Parses a string into a list considering ',', ':' and ';' as delimiters.
    Evaluates each element of the list (strings).

    Parameters
    ----------
    string : str, list
        String to convert into a list

    Returns
    -------
    values : list
        Parsed list
    """
    delimiters = [',', ':', ';']

    for delimiter in delimiters:
        if delimiter in string:
            items = string.split(delimiter)
            break
    else:
        # If no delimiter is found, return the string itself
        return [string]
    # Attempt to evaluate each item; if evaluation fails, keep it as a string
    values = []
    for item in items:
        try:
            values.append(eval(item.strip()))
        except Exception:
            values.append(item.strip())
    return values


def parse_path(path, data_path=None, input_dir=None, version=None, domain_type=None) -> str:
    """
    Parses path using specific patterns.

    This function replaces the following patterns within the input path string with the provided values:
    - '<data_path>'
    - '<input_dir>'
    - '<version>'
    - '<domain_type>'

    Parameters
    ----------
    path : str
        Path with patterns to parse.
    data_path : str, optional
        String to replace the pattern '<data_path>' (default: None).
    input_dir : str, optional
        String to replace the pattern '<input_dir>' (default: None).
    version : str, optional
        String to replace the pattern '<version>' (default: None).
    domain_type : str, optional
        String to replace the pattern '<domain_type>' (default: None).

    Returns
    -------
    path : str
        Parsed path with replaced patterns.
    """

    if data_path is not None:
        path = path.replace("<data_path>", data_path)
    if input_dir is not None:
        path = path.replace("<input_dir>", input_dir)
    if version is not None:
        path = path.replace("<version>", version)
    if domain_type is not None:
        path = path.replace("<domain_type>", domain_type)

    return path


def parse_bool(str_bool):
    """
    Parse the giving string into a boolean.
    The accepted options for a True value are: "True", "true", "T", "t", "Yes", "yes", "Y", "y", "1"
    The accepted options for a False value are: "False", "false", "F", "f", "No", "no", "N", "n", "0"

    If the sting is not in the options it will release a WARNING and the return value will be False.

    Parameters
    ----------
    str_bool : bool or str or int
        Bool to parse
    Returns
    -------
    bool
        Parsed boolean

    """
    true_options = ["True", "true", "T", "t", "Yes", "yes", "Y", "y", "1", 1, True]
    false_options = ["False", "false", "F", "f", "No", "no", "N", "n", "0", 0, False, None, "None", "none"]

    if str_bool in true_options:
        return True
    elif str_bool in false_options:
        return False
    else:
        print("WARNING: Boolean value not contemplated use {0} for True values and {1} for the False ones".format(
            true_options, false_options
        ))
        print("/t Using False as default")
        return False


def parse_start_date(str_date):
    """
    Parse the date form string to datetime.
    It accepts several ways to introduce the date:
        YYYYMMDD, YYYY/MM/DD, YYYYMMDDhh, YYYYYMMDD.hh, YYYY/MM/DD_hh:mm:ss, YYYY-MM-DD_hh:mm:ss,
        YYYY/MM/DD hh:mm:ss, YYYY-MM-DD hh:mm:ss, YYYY/MM/DD_hh, YYYY-MM-DD_hh.

    Parameters
    ----------
    str_date : str or datetime
        Date to parse

    Returns
    -------
    datetime
        Parsed time-step date-time
    """
    format_types = ["%Y%m%d", "%Y%m%d%H", "%Y%m%d.%H", "%Y/%m/%d_%H:%M:%S", "%Y-%m-%d_%H:%M:%S",
                    "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d_%H", "%Y-%m-%d_%H", "%Y/%m/%d"]

    date = None
    for date_format in format_types:
        try:
            date = datetime.strptime(str_date, date_format)
            break
        except ValueError as e:
            if str(e) == "day is out of range for month":
                raise ValueError(e)

    if date is None:
        raise ValueError(f"Date format '{str_date}' not contemplated. Use one of this: {format_types}")

    return date


def parse_float_list(value):
    try:
        # Split the input string into a list of floats
        float_list = [float(item) for item in value.split(",")]
        # Change to numpy array
        float_list = array(float_list, dtype=precision)
        return float_list
    except ValueError:
        raise ArgumentTypeError("Invalid float list format")
