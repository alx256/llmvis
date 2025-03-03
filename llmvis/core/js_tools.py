import re

from typing import Optional


def list_as_js(list: list[any], do_conversion=False) -> str:
    """
    Convert a list to a JavaScript representation of the list.

    Args:
        list (list[any]): The list that should be represented as
            a JavaScript list.
        do_conversion (bool): Setting this to `true` runs an
            associated `get_js` method on each element of the
            provided list when encoding it as a string. Use
            this if each element of the provided list needs a
            special encoding approach for converting to
            JavaScript and has a `get_js` method. `false` will
            ignore this. Default is `false`.

    Returns:
        A string containing the JavaScript representation of this
        list.
    """

    js = "["

    for i, element in enumerate(list):
        js += element.get_js() if do_conversion else str(element)

        if i < len(list) - 1:
            js += ","

    js += "]"
    return js


def escape_all(string: Optional[str]) -> str:
    """
    Given a string, return a new string with necessary special characters
    escaped. Used for escaping strings so that they can safely be
    inserted into JavaScript code stored in a string.

    Args:
        string (str): The string that should necessary special characters
            escaped.

    Returns:
        A new string with necessary special characters escaped.
    """
    return re.sub(r'([\'\n"\\])', r"\\\1", string)
