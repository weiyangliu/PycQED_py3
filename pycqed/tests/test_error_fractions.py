import numpy as np
from unittest import TestCase

from pycqed.analysis import error_fractions as ef


class TestErrorFractions(TestCase):
    def test_conventional_error_fraction(self):
        a = np.ones(5)
        b = np.zeros(5)
        c = [0, 0, 1, 1, 0]
        self.assertEqual(ef.conventional_error_fraction(a, 0), 1)
        self.assertEqual(ef.conventional_error_fraction(b, 0), 0)
        self.assertEqual(ef.conventional_error_fraction(a, 1), 0)
        self.assertEqual(ef.conventional_error_fraction(b, 1), 1)

        self.assertEqual(ef.conventional_error_fraction(c, 0), 0.4)
        self.assertEqual(ef.conventional_error_fraction(c, 1), 0.6)
        with self.assertRaises(ValueError):
            ef.conventional_error_fraction(c, .4)

    def test_restless_error_fraction(self):
        a = np.ones(5)
        b = np.zeros(5)
        c = [0, 1, 0, 1, 0, 1]
        d = [1, 0, 1, 0, 1, 0]
        e = [0, 1, 1, 0, 1, 0]

        # Expected operation 0
        self.assertAlmostEqual(ef.restless_error_fraction(a, 0), 0)
        self.assertAlmostEqual(ef.restless_error_fraction(b, 0), 0)
        self.assertAlmostEqual(ef.restless_error_fraction(c, 0), 1)
        self.assertAlmostEqual(ef.restless_error_fraction(d, 0), 1)
        self.assertAlmostEqual(ef.restless_error_fraction(e, 0), 4/5)

        # Expected operation 1
        self.assertAlmostEqual(ef.restless_error_fraction(a, 1), 1)
        self.assertAlmostEqual(ef.restless_error_fraction(b, 1), 1)
        self.assertAlmostEqual(ef.restless_error_fraction(c, 1), 0)
        self.assertAlmostEqual(ef.restless_error_fraction(d, 1), 0)
        self.assertAlmostEqual(ef.restless_error_fraction(e, 1), 1/5)

        with self.assertRaises(ValueError):
            ef.restless_error_fraction(c, .4)
