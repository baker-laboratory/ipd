import contextlib
import re
from difflib import get_close_matches

import ipd

def get_all_annotations(cls):
    """Get all annotations from a class and its base classes.

    Args:
        cls: The class to retrieve annotations from.

    Returns:
        dict: A dictionary containing the combined annotations.

    Example:
        >>> class Base:
        ...     a: int
        ...
        >>> class Child(Base):
        ...     b: str
        ...
        >>> get_all_annotations(Child)
        {'a': <class 'int'>, 'b': <class 'str'>}
    """
    annotations = {}
    for base in cls.__mro__[::-1]:
        annotations |= getattr(base, '__annotations__', {})
    return annotations

def eval_fstring(template, namespace):
    """Evaluate an f-string template within a given namespace.

    Args:
        template (str): The f-string template.
        namespace (dict): The namespace to use for evaluation.

    Returns:
        str: The evaluated f-string result.

    Example:
        >>> eval_fstring('{x + y}', {'x': 2, 'y': 3})
        '5'
    """
    return eval(f'f"""{template}"""', namespace)
    # return ipd.dev.safe_eval(f'f"""{template}"""', namespace)

def printed_string(thing, rich=True):
    """Capture the printed output of an object as a string.

    Args:
        thing: The object to print.
        rich (bool): Whether to use rich formatting (if available).

    Returns:
        str: The printed output as a string.

    Example:
        >>> printed_string('hello')
        'hello\\n'
    """
    with contextlib.suppress(ImportError):
        if rich:
            from rich import print
    with ipd.dev.capture_stdio() as printed:
        print(thing)
    return printed.read()

def strip_duplicate_spaces(s):
    """Remove duplicate spaces from a string.

    Args:
        s (str): The input string.

    Returns:
        str: The string with duplicate spaces removed.

    Example:
        >>> strip_duplicate_spaces('hello  world')
        'hello world'
    """
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s

def tobytes(s):
    """Convert a string to bytes if it's not already bytes.

    Args:
        s (str | bytes): The input string or bytes.

    Returns:
        bytes: The byte representation of the input.

    Example:
        >>> tobytes('hello')
        b'hello'
        >>> tobytes(b'hello')
        b'hello'
    """
    if isinstance(s, str):
        return s.encode()
    return s

def tostr(s):
    """Convert bytes to a string if it's not already a string.

    Args:
        s (str | bytes): The input string or bytes.

    Returns:
        str: The string representation of the input.

    Example:
        >>> tostr(b'hello')
        'hello'
        >>> tostr('hello')
        'hello'
    """
    if isinstance(s, bytes):
        return s.decode()
    return s

def toname(val):
    """Validate if a string is a valid name (no special characters).

    Args:
        val (str): The input string.

    Returns:
        str | None: The input string if it's valid, otherwise None.

    Example:
        >>> toname('hello')
        'hello'
        >>> toname('hello#')
        """
    if not re.match(r'.*[%^&*#$].*', val):
        return val
    return None

def toidentifier(val):
    """Validate if a string is a valid Python identifier.

    Args:
        val (str): The input string.

    Returns:
        str | None: The input string if it's a valid identifier, otherwise None.

    Example:
        >>> toidentifier('valid_name')
        'valid_name'
        >>> toidentifier('123invalid')
        """
    if isinstance(val, str) and val.isidentifier():
        return val
    return None

def find_close_argnames(word, string_list, n=3, cutoff=0.6):
    """Find close matches to a given word from a list of strings.

    Args:
        word (str): The word to find close matches for.
        string_list (list of str): A list of strings to search within.
        n (int): The maximum number of close matches to return.
        cutoff (float): The minimum similarity score (0-1) for a string to be considered a match.

    Returns:
        list: A list of close matches.

    Example:
        >>> find_close_argnames('apple', ['apple', 'ape', 'apply', 'banana'], n=2)
        ['apple', 'apply']
    """
    candidates = get_close_matches(word, string_list, n=n, cutoff=cutoff)
    candidates = filter(lambda s: abs(len(s) - len(word)) < len(word) // 5, candidates)
    return list(candidates)

ascii_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz0123456789"
ascii_chars += ''.join([chr(i) for i in range(33, 65)])
ascii_chars += ''.join([chr(i) for i in range(91, 97)])
ascii_chars += ''.join([chr(i) for i in range(123, 127)])
