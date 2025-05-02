import difflib
import pytest
from evn.format import MarkHandFormattedBlocksCpp, RuffFormat, CodeFormatter, UnmarkCpp, AlignTokensCpp
import evn

def main():
    evn.testing.quicktest(globals())

splitter = '======== ↑ original ↓ formatted ========'

@pytest.mark.parametrize(
    'testcase',
    [
        """
class Example:                    pass
# fmt: off
class Example:                    pass
# fmt: on
class Example:                    pass
======== ↑ original ↓ formatted ========
class Example:
    pass


# fmt: off
class Example:                    pass
# fmt: on
class Example:
    pass
""",
        """
# fmt: off
class Example: pass
# fmt: on
class Foo: pass
def foo():
    # fmt: off
    if a: b
    else: c
    # fmt: on
    if a: b
    else: c
======== ↑ original ↓ formatted ========
# fmt: off
class Example: pass
# fmt: on
class Foo:
    pass


def foo():
    # fmt: off
    if a: b
    else: c
    # fmt: on
    if a:
        b
    else:
        c
""",
    ],
)
def test_ruff_formatting(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter(actions=[RuffFormat()])
    original, expected = testcase.split(splitter)
    original = original.strip()
    expected = expected.strip()
    formatted = formatter.run({
        'test_case.py': original
    }).buffers['test_case.py']['formatted']
    err = f'Formatting failed on:\n------------------- orig ------------------------\n{original}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected}'
    # print('***************************************')
    # print(expected)
    # print('***************************************')
    # print(formatted)
    # print('***************************************')
    # # print("\n".join(diff))
    # print('\n'.join(difflib.ndiff(expected.splitlines(), formatted.splitlines())))
    # # print(TEST.dev.diff(expected, formatted))
    # print('***************************************')
    assert formatted.strip() == expected.strip(), err


@pytest.mark.parametrize(
    'testcase',
    [
        """
print('hello')
print('world')
======== ↑ original ↓ formatted ========
print('hello')
print('world')
""",
        """
print('start')
def foo(): pass

def bar():
    pass
======== ↑ original ↓ formatted ========
print('start')
#             fmt: off
def foo(): pass
#             fmt: on

def bar():
    pass
""",
        ("""
class Example:
    def method(self):
        if self.flag: return True
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on
    #             fmt: off
    def another(self): return False
    #             fmt: on
"""),
        ("""
class Example:
    def method(self):
        if self.flag: return True
    print('foo')
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    print(bar)
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on
    print('foo')
    #             fmt: off
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    #             fmt: on
    print(bar)
    #             fmt: off
    def another(self): return False
    #             fmt: on
"""),
    ],
)
def test_mark_formatted_blocks(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter(actions=[MarkHandFormattedBlocksCpp()])
    original, expected = testcase.split(splitter)
    original = original.strip()
    expected = expected.strip()
    formatted = formatter.run({
        'test_case.py': original
    }).buffers['test_case.py']['formatted']
    assert (
        formatted.strip() == expected.strip()
    ), f'Formatting failed on:\n------------------- orig ------------------------\n{original}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected}'


@pytest.mark.parametrize(
    'testcase',
    [
        """
print('hello')
print('world')
======== ↑ original ↓ formatted ========
print('hello')
print('world')
""",
        """
import foo
def foo(): pass

def bar():
    pass
======== ↑ original ↓ formatted ========
import foo


#             fmt: off
def foo(): pass
#             fmt: on


def bar():
    pass
""",
        """
class Example:
    def method(self):
        if self.flag: return True
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on

    #             fmt: off
    def another(self): return False
    #             fmt: on

""",
        """
class Example:
    def method(self):
        if self.flag: return True
    print('foo')
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    print(bar)
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on

    print('foo')
    #             fmt: off
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    #             fmt: on
    print(bar)

    #             fmt: off
    def another(self): return False
    #             fmt: on

""",
    ],
)
def test_mark_blocks_ruff(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter(
        actions=[MarkHandFormattedBlocksCpp(),
                 RuffFormat()])
    original, expected = testcase.split(splitter)
    original = original.strip()
    expected = expected.strip()
    formatted = formatter.run({
        'test_case.py': original
    }, debug=False).buffers['test_case.py']['formatted']
    err = f'Formatting failed on:\n------------------- orig ------------------------\n{original}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected}'
    assert formatted.strip() == expected.strip(), err


@pytest.mark.parametrize(
    'testcase',
    [
        """
print('hello')
print('world')
======== ↑ original ↓ formatted ========
print('hello')
print('world')
""",
        """
import foo


#             fmt: off
def foo(): pass
#             fmt: on


def bar():
    pass
======== ↑ original ↓ formatted ========
import foo

def foo(): pass

def bar():
    pass
""",
        """
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on

    #             fmt: off
    def another(self): return False
    #             fmt: on

======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        if self.flag: return True

    def another(self): return False
""",
        """
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on

    print('foo')
    #             fmt: off
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    #             fmt: on
    print(bar)

    #             fmt: off
    def another(self): return False
    #             fmt: on
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        if self.flag: return True

    print('foo')
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    print(bar)

    def another(self): return False
""",
    ],
)
def test_unmark(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter([UnmarkCpp()])
    original, expected = testcase.split(splitter)
    original = original.strip()
    expected = expected.strip()
    formatted = formatter.run({
        'test_case.py': original
    }, debug=False).buffers['test_case.py']['formatted']
    err = f'Formatting failed on:\n------------------- orig ------------------------\n{original}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected}'
    assert formatted.strip() == expected.strip(), err


@pytest.mark.parametrize(
    'testcase',
    [
        """
print('hello')
print('world')
======== ↑ original ↓ formatted ========
print('hello')
print('world')
""",
        """
import foo
def foo(): pass

def bar():
    pass
======== ↑ original ↓ formatted ========
import foo

def foo(): pass

def bar():
    pass
""",
        """
class Example:
    def method(self):
        if self.flag: return True
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        if self.flag: return True

    def another(self): return False
""",
        """
class Example:
    def method(self):
        if self.flag: return True
    print('foo')
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    print(bar)
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        if self.flag: return True

    print('foo')
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta,    epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    print(bar)

    def another(self): return False
""",
    ],
)
def test_mark_blocks_ruff_unmark(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter(
        [MarkHandFormattedBlocksCpp(),
         RuffFormat(), UnmarkCpp()])
    original, expected = testcase.split(splitter)
    original = original.strip()
    expected = expected.strip()
    formatted = formatter.run({
        'test_case.py': original
    }, debug=False).buffers['test_case.py']['formatted']
    err = f'Formatting failed on:\n------------------- orig ------------------------\n{original}\n------------------------ Got:------------------------\n{formatted}\n------------------------Expected: ------------------------\n{expected}'
    assert formatted.strip() == expected.strip(), err


@pytest.mark.parametrize(
    'testcase',
    [
        """
print('hello')
print('world')
======== ↑ original ↓ formatted ========
#             fmt: off
print('hello')
print('world')
#             fmt: on
""",
        """
import foo
def foo(): pass
def bar():
    pass
======== ↑ original ↓ formatted ========
import foo
#             fmt: off
def foo(): pass
#             fmt: on
def bar():
    pass
""",
        """
class Example:
    def method(self):
        if self.flag: return True
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on
    #             fmt: off
    def another(self): return False
    #             fmt: on
""",
        """
class Example:
    def method(self):
        if self.flag: return True
    print('foo')
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta, gamma, delta, epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta, gamma, delta, epsilon,]
    monkey = [banana, kiwi , apple, strawberry,]
    print(bar)
    def another(self): return False
======== ↑ original ↓ formatted ========
class Example:
    def method(self):
        #             fmt: off
        if self.flag: return True
        #             fmt: on
    print('foo')
    #             fmt: off
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta, epsilon   ,]
    monkey = [banana, kiwi , apple, strawberry,]
    alpha  = [beta  , gamma, delta, epsilon   ,]
    monkey = [banana, kiwi , apple, strawberry,]
    #             fmt: on
    print(bar)
    #             fmt: off
    def another(self): return False
    #             fmt: on
""",
    ],
)
def test_cpp_align_tokens(testcase):
    """Test full formatting pipeline: AddFmtMarkers → RuffFormat → RemoveFmtMarkers → RemoveExtraBlankLines."""
    formatter = CodeFormatter([AlignTokensCpp()])
    original, expected = testcase.split(splitter)
    original = original.strip()
    expected = expected.strip()
    formatted = formatter.run({
        'test_case.py': original
    }, debug=False).buffers['test_case.py']['formatted']
    err = '\n'.join(
        difflib.ndiff(expected.splitlines(), formatted.splitlines()))
    assert formatted.strip() == expected.strip(), err


if __name__ == '__main__':
    main()
