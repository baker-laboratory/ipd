import unittest
from evn import isstr, isint, islist, isdict, isseq, ismap, isseqmut, ismapmut, isiter

import evn

config_test = evn.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)


def main():
    evn.testing.quicktest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )


class TestTypeCheckers(unittest.TestCase):

    def test_isstr(self):
        self.assertTrue(isstr('hello'))
        self.assertFalse(isstr(123))
        self.assertFalse(isstr([]))

    def test_isint(self):
        self.assertTrue(isint(42))
        self.assertFalse(isint('42'))
        self.assertFalse(isint(3.14))

    def test_islist(self):
        self.assertTrue(islist([1, 2, 3]))
        self.assertFalse(islist((1, 2, 3)))
        self.assertFalse(islist('list'))

    def test_isdict(self):
        self.assertTrue(isdict({'key': 'value'}))
        self.assertFalse(isdict([('key', 'value')]))
        self.assertFalse(isdict('dict'))

    def test_isseq(self):
        self.assertTrue(isseq([1, 2, 3]))
        self.assertTrue(isseq((1, 2, 3)))
        self.assertTrue(isseq('sequence'))
        self.assertFalse(isseq(42))

    def test_ismap(self):
        self.assertTrue(ismap({'key': 'value'}))
        self.assertFalse(ismap([('key', 'value')]))
        self.assertFalse(ismap('map'))

    def test_isseqmut(self):
        self.assertTrue(isseqmut([1, 2, 3]))
        self.assertFalse(isseqmut((1, 2, 3)))  # Tuples are not mutable
        self.assertFalse(isseqmut('string'))  # Strings are immutable

    def test_ismapmut(self):
        self.assertTrue(ismapmut({'key': 'value'}))
        self.assertFalse(ismapmut(frozenset([('key', 'value')
                                             ])))  # frozenset is immutable
        self.assertFalse(ismapmut('map'))

    def test_isiter(self):
        self.assertTrue(isiter([1, 2, 3]))
        self.assertTrue(isiter((1, 2, 3)))
        self.assertTrue(isiter('iterable'))
        self.assertTrue(isiter({1, 2, 3}))
        self.assertFalse(isiter(42))


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    main()
