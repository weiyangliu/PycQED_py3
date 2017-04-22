import unittest
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma


class Test_J2_chevron_analysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir
        self.ma_obj = ma.Chevron_2D(label='chevron_tau1')

    def test_effective_coupling(self):
        pass
