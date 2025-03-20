import pytest
import ipd
from ipd.dev.code.format_code import (AddFmtMarkers, RuffFormat, RemoveFmtMarkers, CodeFormatter)

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        check_xfail=False,
    )

from ipd.dev.code.format_code import (RemoveExtraBlankLines)

@pytest.mark.parametrize(
    "testcase",
    [
        """
# fmt: off
class Example:
# fmt: on
# fmt: off
    pass
# fmt: on
======== ↑ original ↓ formatted ========
# fmt: off
class Example:
# fmt: on
# fmt: off
    pass
# fmt: on
"""
    ],
)
def test_ruff_formatting(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter(actions=[RuffFormat()])
    original_code, expected_code = testcase.split("======== ↑ original ↓ formatted ========")
    formatted = formatter.run({"test_case.py": original_code}).buffers["test_case.py"]["formatted"]
    assert formatted.strip() == expected_code.strip(
    ), f"Formatting failed on:\n------------------- orig ------------------------\n{original_code}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected_code}"

@pytest.mark.parametrize(
    "testcase",
    [
        ("""print('hello')




print('world')
======== ↑ original ↓ formatted ========
print('hello')

print('world')"""),
        ("""def foo(): pass

def bar():
    pass
======== ↑ original ↓ formatted ========
def foo(): pass

def bar():
    pass"""),
        ("""
class Example:
    def method(self):
        if self.flag: return True
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        if self.flag: return True

    def another(self): return False"""),
    ],
)
def test_code_formatting(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter(
        actions=[AddFmtMarkers(), RuffFormat(
        ), RemoveFmtMarkers(), RemoveExtraBlankLines()])
    original_code, expected_code = testcase.split("======== ↑ original ↓ formatted ========")
    formatted = formatter.run({"test_case.py": original_code}).buffers["test_case.py"]["formatted"]
    assert formatted.strip() == expected_code.strip(
    ), f"Formatting failed on:\n------------------- orig ------------------------\n{original_code}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected_code}"

if __name__ == '__main__':
    main()
