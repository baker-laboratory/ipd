import importlib.util
from typing import TypeVar

T = TypeVar('T')


def is_installed(name):
    return importlib.util.find_spec(name)


def not_installed(name):
    return not importlib.util.find_spec(name)
