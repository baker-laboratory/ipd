from evn.tree.tree_diff import (
    TreeDiffer,
    SummaryStrategy,
    IgnoreKeyStrategy,
    IgnoreUnderscoreKeysStrategy,
    FloatToleranceStrategy,
    ShallowOnPathStrategy,
    SummaryOnPathStrategy,
)


def test_scalar_difference():
    cfg1 = {'a': 1}
    cfg2 = {'a': 2}
    result = TreeDiffer().diff(cfg1, cfg2)
    assert result == {'a': (1, 2)}


def test_nested_difference():
    cfg1 = {'a': {'b': {'c': 1}}}
    cfg2 = {'a': {'b': {'c': 2}}}
    result = TreeDiffer().diff(cfg1, cfg2)
    assert result == {'a': {'b': {'c': (1, 2)}}}


def test_missing_keys():
    cfg1 = {'a': 1, 'b': 2}
    cfg2 = {'a': 1, 'c': 3}
    result = TreeDiffer(summarize_subtrees=True).diff(cfg1, cfg2)
    assert isinstance(result, tuple)
    assert "'b': 2" in result[0]
    assert "'c': 3" in result[1]


def test_list_flat_and_nested():
    cfg1 = {'x': [1, {'a': 'old'}, 3]}
    cfg2 = {'x': [1, {'a': 'new'}, 4]}
    result = TreeDiffer().diff(cfg1, cfg2)
    assert result == {
        'x': {
            '_flat': ([3], [4]),
            '_nested': {
                1: {
                    'a': ('old', 'new')
                }
            }
        }
    }


def test_flattened_output():
    cfg1 = {'a': {'b': {'c': 1}}, 'x': [1, 2]}
    cfg2 = {'a': {'b': {'c': 2}}, 'x': [1, 3]}
    differ = TreeDiffer(flatpaths=True)
    result = differ.diff(cfg1, cfg2)
    assert 'a.b.c' in result
    assert 'x:_flat' in result


def test_summary_strategy():
    cfg1 = {'a': {'x': 1, 'y': 2}}
    cfg2 = {'a': {'x': 3, 'z': 4}}
    differ = TreeDiffer(strategy=SummaryStrategy(), summarize_subtrees=True)
    result = differ.diff(cfg1, cfg2)
    assert isinstance(result, tuple)
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


def test_ignore_keys():
    cfg1 = {'a': 1, '_meta': 123}
    cfg2 = {'a': 2, '_meta': 456}
    differ = TreeDiffer(strategy=IgnoreKeyStrategy())
    result = differ.diff(cfg1, cfg2)
    assert result == {'a': (1, 2)}


def test_list_dicts_zip_match():
    cfg1 = {'lst': [{'x': 1}, {'y': 2}]}
    cfg2 = {'lst': [{'x': 1}, {'y': 3}]}
    result = TreeDiffer().diff(cfg1, cfg2)
    assert result == {'lst': {'_nested': {1: {'y': (2, 3)}}}}


def test_cycle_detection():
    a, b = {}, {}
    a['self'] = a
    b['self'] = b
    differ = TreeDiffer()
    result = differ.diff(a, b)
    assert result is None


def test_max_depth():
    cfg1 = {'a': {'b': {'c': 1}}}
    cfg2 = {'a': {'b': {'c': 2}}}
    result = TreeDiffer(max_depth=2).diff(cfg1, cfg2)
    assert result is None
    cfg1 = {'a': {'b': {'d': 1}}}
    cfg2 = {'a': {'b': {'c': 2}}}
    result = TreeDiffer(max_depth=2, flatpaths=True).diff(cfg1, cfg2)
    assert result == {'a.b': ({'d': 1}, {'c': 2})}


def test_custom_path_fmt():
    cfg1 = {'a': {'b': 1}}
    cfg2 = {'a': {'b': 2}}
    differ = TreeDiffer(flatpaths=True, path_fmt=lambda p: '/'.join(p))
    result = differ.diff(cfg1, cfg2)
    assert result == {'a/b': (1, 2)}


# --- Strategy Subclasses for Testing ---
# --- Feature Combination Tests ---


def test_ignore_keys_and_nested_diff():
    cfg1 = {'a': 1, '_meta': 999, 'b': {'x': 1}}
    cfg2 = {'a': 2, '_meta': 1000, 'b': {'x': 1}}
    differ = TreeDiffer(strategy=IgnoreUnderscoreKeysStrategy())
    result = differ.diff(cfg1, cfg2)
    assert result == {'a': (1, 2)}


def test_float_tolerance_in_list_flat_diff():
    cfg1 = {'nums': [1.0, 2.0000001]}
    cfg2 = {'nums': [1.0, 2.0]}
    differ = TreeDiffer(strategy=FloatToleranceStrategy())
    result = differ.diff(cfg1, cfg2)
    assert result is None


def test_shallow_on_path_strategy():
    cfg1 = {'meta': {'version': 1}, 'real': {'value': 10}}
    cfg2 = {'meta': {'version': 2}, 'real': {'value': 20}}
    differ = TreeDiffer(strategy=ShallowOnPathStrategy())
    result = differ.diff(cfg1, cfg2)
    assert result == {
        'meta': ({
            'version': 1
        }, {
            'version': 2
        }),
        'real': {
            'value': (10, 20)
        }
    }


def test_summary_on_path_strategy():
    cfg1 = {'data': {'big': list(range(1000))}, 'other': 1}
    cfg2 = {'data': {'big': list(range(999)) + [9999]}, 'other': 2}
    differ = TreeDiffer(strategy=SummaryOnPathStrategy(),
                        summarize_subtrees=True)
    result = TreeDiffer(strategy=SummaryOnPathStrategy(),
                        summarize_subtrees=True).diff(cfg1, cfg2)
    assert isinstance(result['data'][0], str)
    assert '2, 13, 14, 15, 16, 17' in result['data'][1]
    assert result['other'] == (1, 2)


def test_flat_output_with_summary_and_ignore():

    class ComboStrategy(IgnoreUnderscoreKeysStrategy, SummaryOnPathStrategy):
        pass

    cfg1 = {'info': {'data': [1, 2, 3]}, '_meta': 'skip'}
    cfg2 = {'info': {'data': [1, 2, 999]}, '_meta': 'also skip'}

    differ = TreeDiffer(strategy=ComboStrategy(),
                        flatpaths=True,
                        summarize_subtrees=True)
    result = differ.diff(cfg1, cfg2)

    assert 'info.data:_diff' in result or 'info.data' in result
    assert '_meta' not in result
