import unittest
import os
import pygsti
import pycqed as pq

class Test_GST(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        fp = os.path.join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                          '_pygsti_Gatesequences', '1Q_5prim_germs.txt')
        self.germs = pygsti.io.load_gatestring_list(fp)

    def test_tomo_analysis_cardinal_state(self):
        self.germs
        # res = ma.Tomo_Multiplexed(label='Tomo_{}'.format(31),
        #                           target_cardinal=None,
        #                           MLE=False)
        # res = ma.Tomo_Multiplexed(label='Tomo_{}'.format(31),
        #                           target_cardinal=31,
        #                           MLE=True)
