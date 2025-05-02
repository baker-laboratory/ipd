import contextlib
import os
import sys
import tempfile
import shutil
from typing import Tuple, Generator
from pathlib import Path

import pytest

import evn
from evn._prelude.import_util import is_installed


def main():
        evn.testing.quicktest(globals())

def test_is_installed():
    assert is_installed('icecream')
    assert not is_installed('lwySENESIONAOIRENTeives')


if __name__ == '__main__':
    main()
