cfg1 = {
    'a': 1,
    'b': [1, 2],
    'c': {
        'x': 10,
        'y': 20
    },
    'd': 'hello',
    'e': {
        'shared': 5,
        'removed_subtree': {
            'k': 9
        }
    },
}

cfg2 = {
    'a': 1,
    'b': [2, 3],
    'c': {
        'x': 10,
        'z': 30
    },
    'd': 123,
    'e': {
        'shared': 6,
        'added_subtree': {
            'new': True
        }
    },
}

diff = dict(
    b=([1], [3]),
    c=({
        'y': 20
    }, {
        'z': 30
    }),
    d=('hello', 123),
    e=({
        'removed_subtree': {
            'k': 9
        }
    }, {
        'added_subtree': {
            'new': True
        }
    }),
)

e = {
    '_diff': ({
        'removed_subtree': {
            'k': 9
        }
    }, {
        'added_subtree': {
            'new': True
        }
    }),
    'common_subtree1': '<more diffs>',
    'common_subtree2': 'even more diffs',
}
