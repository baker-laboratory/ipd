import re


def extract_param_help(docstring: str) -> dict[str, str]:
    """
    Extract parameter descriptions from a Sphinx-style docstring.

    :param docstring: The docstring to parse.
    :type docstring: str
    :return: Mapping from parameter name to help text.
    :rtype: dict[str, str]
    """
    if not docstring:
        return {}

    param_help = {}
    pattern = re.compile(r'^\s*:param (\w+)\s*:\s*(.+)$', re.MULTILINE)

    for match in pattern.finditer(docstring):
        name, desc = match.groups()
        param_help[name] = desc.strip()

    return param_help
