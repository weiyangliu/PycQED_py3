import numpy as np
from unittest import TestCase

from pycqed.simulations import chevron_sim as chs
from pycqed.instrument_drivers.meta_instrument.kernel_object import (
    bounce_kernel, decay_kernel, skin_kernel, poly_kernel)


class TestChevronSim(TestCase):

    @classmethod
    def setUpClass(self):
        self.e_min = -0.0322
        self.e_max = 0.0322
        self.e_points = 20
        self.time_stop = 60
        self.time_step = 4
        self.bias_tee = lambda self, t, a, b, c: t*a**2+t*b+c

        # self.distortion = lambda t: self.lowpass_s(t, 2)
        self.distortion = lambda self, t: self.bias_tee(t, 0., 2e-5, 1)

        self.time_vec = np.arange(0., self.time_stop, self.time_step)
        self.freq_vec = np.linspace(self.e_min, self.e_max, self.e_points)

    def test_output_shape_chevron_sim(self):
        """
        Trivial test that just checks if there is nothing that broke the
        chevron sims in a way that breaks it.
        """

        result = chs.chevron(2.*np.pi*(6.552 - 4.8),
                             self.e_min, self.e_max,
                             self.e_points,
                             np.pi*0.0385,
                             self.time_stop,
                             self.time_step,
                             self.distortion)
        self.assertEqual(np.shape(result),
                         (len(self.freq_vec), len(self.time_vec)+1))


class TestKernelObj(TestCase):

    @classmethod
    def setUpClass(self):
        pass

    def test_bounce_kernel(self):
        result = bounce_kernel(amp=0.02, time=4, length=20)
        expected_result = np.array([0.98039216,  0.,  0.,  0.,  0.01960784,
                                    0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.])
        np.testing.assert_array_almost_equal(result, expected_result)
        self.assertEqual(np.sum(result), 1)

    def test_decay_kernel(self):
        result = decay_kernel(amp=.9, tau=10, length=400)
        expected_result = np.array([1.9, -0.08564632, -0.077496,
                                    -0.07012128, -0.06344836, -0.05741045,
                                    -0.05194712, -0.0470037, -0.04253071,
                                    -0.03848337])
        np.testing.assert_array_almost_equal(result[0:10], expected_result)
        self.assertEqual(np.sum(result), 1)

    def test_skin_kernel(self):
        result = skin_kernel(alpha=.5, length=300)
        expected_result = np.array(
            [9.74650704e-01,   7.87785755e-03,   3.49065723e-03,
             2.08099100e-03,   1.42019093e-03,   1.04836603e-03,
             8.14802756e-04,   6.56801328e-04,   5.44043050e-04,
             4.60247503e-04])
        np.testing.assert_array_almost_equal(result[0:10], expected_result)
        self.assertEqual(np.sum(result), 1)

    def test_poly_kernel(self):
        result = poly_kernel(a=1, b=300, c=2, length=40)
        expected_result = np.array(
            [2.,  301.,  303.,  305.,  307.,  309.,  311.,  313.,  315.,
                317.,  319.,  321.,  323.,  325.,  327.,  329.,  331.,  333.,
                335.,  337.,  339.,  341.,  343.,  345.,  347.,  349.,  351.,
                353.,  355.,  357.,  359.,  361.,  363.,  365.,  367.,  369.,
                371.,  373.,  375.,  377.])
        np.testing.assert_array_almost_equal(result, expected_result)
