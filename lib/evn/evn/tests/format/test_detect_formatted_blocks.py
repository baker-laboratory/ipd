import pytest
import evn


def main():
    pass


@pytest.fixture
def ifb():
    return evn.format.IdentifyFormattedBlocks()


def test_strange_cases(ifb):
    assert ifb.mark_formtted_blocks('', threshold=0.99) == ''
    assert ifb.mark_formtted_blocks('\n', threshold=0.99) == '\n'
    assert ifb.mark_formtted_blocks('\n\n', threshold=0.99) == '\n\n'
    assert ifb.mark_formtted_blocks('========', threshold=0.99)
    # assert ifb.mark_formtted_blocks('========\n========', threshold=0.99)
    # assert ifb.mark_formtted_blocks('========\n========\n', threshold=0.99)
    assert ifb.unmark('') == ''
    assert ifb.unmark('\n') == '\n'
    assert ifb.unmark('\n\n') == '\n'
    assert ifb.unmark('========')
    assert ifb.unmark('========\n========')
    assert ifb.unmark('========\n========\n')


def test_unmark(ifb):
    test = """q
    #             fmt: on
z
    #             fmt: off
a
#             fmt: off
b
#             fmt: on
c
"""
    test2 = """q
z
a
b
c
"""
    assert ifb.unmark(test) == test2


def test_mark_formtted_blocks_no_change(ifb):
    # If lines are dissimilar (or threshold is set high),
    # the mark_formtted_blocks function should return code without formatting markers.
    code = 'line one\nline two'
    result = ifb.mark_formtted_blocks(code, threshold=5)
    # We strip both strings to avoid any trailing newline differences.
    assert result.strip() == code.strip()


def test_mark_formtted_blocks_formatting(ifb):
    # When lines are similar, formatting markers should be inserted.
    # Here we use a low threshold to force detection.
    code = '\n    int a = 0;\n    int a = 0;'
    print(ifb.compute_similarity_score('int a = 0', 'int a = 0'))
    result = ifb.mark_formtted_blocks(code, threshold=2)
    assert '#             fmt: off' in result
    assert '#             fmt: on' in result


def test_whitespace(ifb):
    # Test that the function handles leading/trailing whitespace correctly.
    code = '    line one\n\n\n\n    line two'
    result = ifb.unmark(code)
    assert len(result.split('\n')) == 4


def test_inline_blocks_are_marked(ifb):
    # Test that inline blocks are marked correctly.
    code = """
    def example_function(): foo
        if True: False
            class Banana: ...
    elif bar: baz
        else: qux
"""
    result = ifb.mark_formtted_blocks(code, threshold=2)
    assert (result == """
    #             fmt: off
    def example_function(): foo
    #             fmt: on
        #             fmt: off
        if True: False
        #             fmt: on
            #             fmt: off
            class Banana: ...
            #             fmt: on
    #             fmt: off
    elif bar: baz
    #             fmt: on
        #             fmt: off
        else: qux
        #             fmt: on
""")


def test_inline_blocks_are_marked2(ifb):
    # Test that inline blocks are marked correctly.
    code = """
    print('foo')

    def example_function(): foo
        dummy
        if True: False

            dummy2
            class Banana: ...
    elif bar: baz
        else: qux
    aaaa
"""
    result = ifb.mark_formtted_blocks(code, threshold=2)
    assert (result == """
    print('foo')

    #             fmt: off
    def example_function(): foo
    #             fmt: on
        dummy
        #             fmt: off
        if True: False
        #             fmt: on

            dummy2
            #             fmt: off
            class Banana: ...
            #             fmt: on
    #             fmt: off
    elif bar: baz
    #             fmt: on
        #             fmt: off
        else: qux
        #             fmt: on
    aaaa
""")


def test_multiline_is_ignored(ifb):
    # Test that inline blocks are marked correctly.
    code = """
    if a: return b \\
    else: return c
    for a in b: c
"""
    result = ifb.mark_formtted_blocks(code, threshold=4)
    assert (result == """
    if a: return b \\
    else: return c
    #             fmt: off
    for a in b: c
    #             fmt: on
""")


if __name__ == '__main__':
    main()
