import difflib
import pytest
import evn

# --- Tokenization Tests ---


@pytest.fixture
def tokenizer():
    return evn.format.PythonLineTokenizer()


def helper_test_reformat_lines(tokenizer, lines, expected):
    output = tokenizer.reformat_lines(lines, add_fmt_tag=True)
    if output == expected:
        return
    output = tokenizer.reformat_lines(lines, add_fmt_tag=True, debug=True)
    print('+++++++++++++++++++ ndiff +++++++++++++++++++')
    print('\n'.join(difflib.ndiff(expected, output)))
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    assert output == expected


def test_tokenize_basic(tokenizer):
    code_line = 'a = b + c'
    tokens = evn.format.tokenize(code_line)
    expected = ['a', '=', 'b', '+', 'c']
    assert tokens == expected


def test_tokenize_with_comment(tokenizer):
    code_line = 'x = 42  # this is a comment'
    tokens = evn.format.tokenize(code_line)
    expected = ['x', '=', '42', '# this is a comment']
    assert tokens == expected


def test_tokenize_string_literal(tokenizer):
    code_line = "print('Hello, \\'World\\'!')"
    tokens = evn.format.tokenize(code_line)
    expected = ['print', '(', "'Hello, \\'World\\'!'", ')']
    assert tokens == expected


def test_tokenize_fstring(tokenizer):
    code_line = 'f"Hello, {name}!"'
    tokens = evn.format.tokenize(code_line)
    expected = ['f"Hello, {name}!"']
    assert tokens == expected


def test_tokenize_triple_quote_string(tokenizer):
    code_line = "s = '''string literal'''"
    tokens = evn.format.tokenize(code_line)
    expected = ['s', '=', "'''string literal'''"]
    assert tokens == expected


def test_tokenize_lambda_expression(tokenizer):
    code_line = 'lambda x, y=42: (x + y) if x > y else (x - y)'
    tokens = evn.format.tokenize(code_line)
    expected = [
        'lambda',
        'x',
        ',',
        'y',
        '=',
        '42',
        ':',
        '(',
        'x',
        '+',
        'y',
        ')',
        'if',
        'x',
        '>',
        'y',
        'else',
        '(',
        'x',
        '-',
        'y',
        ')',
    ]
    assert tokens == expected
    assert tokenizer.join_tokens(tokens) == code_line


def test_tokenize_multi_char_operators(tokenizer):
    code_line = 'a ** b // c != d -> e'
    tokens = evn.format.tokenize(code_line)
    expected = ['a', '**', 'b', '//', 'c', '!=', 'd', '->', 'e']
    assert tokens == expected


def test_tokenize_while(tokenizer):
    code_line = 'while if this is True: break out # comment'
    tokens = evn.format.tokenize(code_line)
    expected = [
        'while', 'if', 'this', 'is', 'True', ':', 'break', 'out', '# comment'
    ]
    assert tokens == expected
    assert not evn.format.is_oneline_statement(tokens[:-3])
    assert evn.format.is_oneline_statement(tokens[:-2])
    assert evn.format.is_oneline_statement(tokens)


# --- Join Tokens (Black-like formatting) Tests ---


def test_join_tokens_default_black_formatting(tokenizer):
    tokens = ['def', 'greet', '(', 'name', ')', ':']
    joined = tokenizer.join_tokens(tokens)
    # Expected: no space between function name and its call parentheses.
    expected = 'def greet(name):'
    assert joined == expected


def test_join_tokens_with_operators(tokenizer):
    tokens = ['a', '=', 'b', '+', 'c', '*', '(', 'd', '-', 'e', ')', '/', 'f']
    joined = tokenizer.join_tokens(tokens)
    expected = 'a = b + c * (d - e) / f'
    assert joined == expected


def test_join_tokens_with_commas_and_colons(tokenizer):
    tokens = ['print', '(', "'a'", ',', "'b'", ',', "'c'", ')']
    joined = tokenizer.join_tokens(tokens)
    expected = "print('a', 'b', 'c')"
    assert joined == expected


def test_join_tokens_complex_expression(tokenizer):
    code_line = 'def func(a, b=2): return a**b + (a - b)*3.14'
    tokens = evn.format.tokenize(code_line)
    joined = tokenizer.join_tokens(tokens)
    expected = 'def func(a, b=2): return a ** b + (a - b) * 3.14'
    assert joined == expected


def test_join_tokens_operator_spacing(tokenizer):
    tokens = ['x', '=', 'a', '==', 'b', 'and', 'c', '!=', 'd']
    joined = tokenizer.join_tokens(tokens)
    expected = 'x = a == b and c != d'
    assert joined == expected


def test_join_tokens_nested_parens(tokenizer):
    code_line = 'print((a+b)*(c-d))'
    tokens = evn.format.tokenize(code_line)
    joined = tokenizer.join_tokens(tokens)
    expected = 'print((a+b) * (c-d))'
    assert joined == expected


def test_join_tokens_mixed_syntax(tokenizer):
    code_line = 'def add(a,b):return a+b'
    tokens = evn.format.tokenize(code_line)
    joined = tokenizer.join_tokens(tokens)
    expected = 'def add(a, b): return a + b'
    assert joined == expected


# --- Token Matching Tests ---


def test_tokens_match_wildcards(tokenizer):
    code1 = 'def compute(x): return 100 + x'
    code2 = 'def compute(y): return 200 + y'
    tokens1 = evn.format.tokenize(code1)
    tokens2 = evn.format.tokenize(code2)
    assert evn.format.tokens_match(tokens1, tokens2)


def test_tokens_match_fail_due_to_operator(tokenizer):
    code1 = 'def compute(x): return 100 * x'
    code2 = 'def compute(x): return 100 / x'
    tokens1 = evn.format.tokenize(code1)
    tokens2 = evn.format.tokenize(code2)
    assert not evn.format.tokens_match(tokens1, tokens2)


def test_tokens_match_keyword_mismatch(tokenizer):
    code1 = 'if x > 0: print(x)'
    code2 = 'while x > 0: print(x)'
    tokens1 = evn.format.tokenize(code1)
    tokens2 = evn.format.tokenize(code2)
    assert not evn.format.tokens_match(tokens1, tokens2)


def test_tokens_match_identifier_vs_string(tokenizer):
    code1 = "def f(x): return 'value'"
    code2 = "def f('x'): return x"
    tokens1 = evn.format.tokenize(code1)
    tokens2 = evn.format.tokenize(code2)
    assert not evn.format.tokens_match(tokens1, tokens2)


def test_no_blocks(tokenizer):
    # All lines have different token patterns; output should match input exactly.
    lines = [
        "print('foo')",
        'b = 2',
        'if a > b: print(a)',
        'while True: break',
    ]
    expected = [
        "print('foo')",
        'b = 2',
        '#             fmt: off',
        'if a > b: print(a)',
        '#             fmt: on',
        '#             fmt: off',
        'while True: break',
        '#             fmt: on',
    ]
    helper_test_reformat_lines(tokenizer, lines, expected)


def test_single_block_alignment(tokenizer):
    # All lines share the same token pattern and indentation.
    # For example, three assignment lines.
    lines = ['    a=1', '    bb=22', '    ccc=333']
    output = tokenizer.reformat_lines(lines)
    # Instead of splitting on whitespace, we check that the '=' appears at the same index.
    eq_indices = [line.find('=') for line in output if '=' in line]
    assert len(set(eq_indices)
               ) == 1, f'Equal sign appears at different columns: {eq_indices}'


def test_multiple_blocks(tokenizer):
    # Mix several blocks.
    lines = [
        'x=10', 'y=20', 'z=30', '', '    a=1', '    bb=22', '    ccc=333',
        "print('done')"
    ]
    output = tokenizer.reformat_lines(lines)
    # For the first block (no indent), check that the '=' sign is in the same column.
    block1 = [
        line for line in output
        if not line.startswith('    ') and line.strip() and 'print' not in line
    ]
    if len(block1) >= 3:
        indices1 = [line.find('=') for line in block1 if '=' in line]
        assert len(set(indices1)) == 1, f"Block1 '=' indices: {indices1}"
    # For the second block (indent 4 spaces)
    block2 = [line for line in output if line.startswith('    ')]
    if len(block2) >= 3:
        indices2 = [line.find('=') for line in block2 if '=' in line]
        assert len(set(indices2)) == 1, f"Block2 '=' indices: {indices2}"


def test_block_boundary_by_indent(tokenizer):
    # Two blocks with different indentations.
    lines = ['def f():', '    a=1', '    b=2', '    c=3', "print('done')"]
    output = tokenizer.reformat_lines(lines)
    # The block inside the function should be aligned, while the other lines are unchanged.
    assert output[0] == 'def f():'
    assert output[-1] == "print('done')"
    inner = [line for line in output if line.startswith('    ')]
    eq_indices = [line.find('=') for line in inner if '=' in line]
    assert len(set(eq_indices)) == 1, f"Inner block '=' indices: {eq_indices}"


def test_block_boundary_by_token_pattern(tokenizer):
    # Lines with same indentation but different token patterns should not group.
    lines = ['    a=1', '    print(a)', '    b=2']
    output = tokenizer.reformat_lines(lines)
    # Expect that no block is formed; output should equal input.
    assert output == lines


def test_heuristic_length_mismatch(tokenizer):
    # If one line is very short and the next is much longer, they should not be grouped.
    lines = ['    a=1', '    verylongidentifier=22']
    output = tokenizer.reformat_lines(lines)
    # Expect that each line is output separately.
    assert output == lines


def test_empty_and_whitespace_lines(tokenizer):
    # Blank lines or lines with only spaces should be preserved.
    lines = ['    a=1', '    b=2', '    ', '', '    c=3']
    output = tokenizer.reformat_lines(lines)
    assert output[2] == ''
    assert output[3] == ''


def test_multiple_blocks_combined(tokenizer):
    # Mix several blocks with different indents and token patterns.
    lines = [
        'x=10', 'y=20', 'z=30', '', '    a=1', '    bb=22', '    ccc=333',
        "print('done')"
    ]
    output = tokenizer.reformat_lines(lines)
    # Check that each block is independently formatted.
    block1 = [
        line for line in output
        if not line.startswith('    ') and line.strip() and 'print' not in line
    ]
    block2 = [line for line in output if line.startswith('    ')]
    if len(block1) >= 3:
        indices1 = [line.find('=') for line in block1 if '=' in line]
        assert len(set(indices1)) == 1, f"Block1 '=' indices: {indices1}"
    if len(block2) >= 3:
        indices2 = [line.find('=') for line in block2 if '=' in line]
        assert len(set(indices2)) == 1, f"Block2 '=' indices: {indices2}"


def test_reformat_lines_with_comments(tokenizer):
    # Ensure comments are preserved and not altered.
    lines = ['x=10  # This is a comment', 'y=20', 'z=30  # Another comment']
    output = tokenizer.reformat_lines(lines)
    assert output[0] == 'x=10  # This is a comment'
    assert output[1] == 'y=20'
    assert output[2] == 'z=30  # Another comment'


def test_reformat_lines_with_empty_lines(tokenizer):
    # Ensure empty lines are preserved.
    lines = ['x=10', '', 'y=20', '  ', 'z=30']
    output = tokenizer.reformat_lines(lines)
    assert output[1] == ''
    assert output[3] == ''


def test_reformat_lines_with_mixed_indentation(tokenizer):
    # Ensure mixed indentation is preserved.
    lines = ['x=10', '    y=20', 'z=30', '    ', '  a=1']
    output = tokenizer.reformat_lines(lines)
    assert output[1] == '    y=20'
    assert output[3] == ''
    assert output[4] == '  a=1'
