import logging
import numpy as np

# This try except statement exists to allow testing without qcodes installed.
# this try except loop should be removed when qcodes is publicly
try:
    from qcodes.instrument.base import Instrument
    from qcodes.utils import validators as vals
    from qcodes.instrument.parameter import ManualParameter
except ImportError:
    pass

from pycqed.measurement.kernel_functions import (
    bounce_kernel,  decay_kernel, skin_kernel, poly_kernel)
from pycqed.measurement.kernel_functions_old import (
    kernel_generic, htilde_bounce, htilde_skineffect,
    save_kernel, step_bounce, step_skineffect)


class Distortion(Instrument):

    '''
    Implements a distortion kernel for a flux channel.
    It contains the parameters and functions needed to produce a kernel file
    according to the models shown in the functions.
    '''
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.add_parameter('skineffect_alpha', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('skineffect_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_amp', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_tau', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('decay_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('bounce_amp', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('bounce_tau', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('bounce_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_a', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_b', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_c', units='',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('poly_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())
        self.add_parameter('corrections_length', units='ns',
                           parameter_class=ManualParameter,
                           vals=vals.Numbers())

    def get_bounce_kernel(self):
        return bounce_kernel(amp=self.bounce_amp(), time=self.bounce_tau(),
                             length=self.bounce_length())

    def get_skin_kernel(self):
        return skin_kernel(alpha=self.skineffect_alpha(),
                           length=self.skineffect_length())

    def get_decay_kernel(self):
        return decay_kernel(amp=self.decay_amp(), tau=self.decay_tau(),
                            length=self.decay_length())

    def get_poly_kernel(self):
        return poly_kernel(a=self.poly_a(),
                           b=self.poly_b(),
                           c=self.poly_c(),
                           length=self.poly_length())

    def convolve_kernel(self, kernel_list, length=None):
        if length is None:
            length = max([len(k) for k in kernel_list])
        total_kernel = kernel_list[0]
        for k in kernel_list[1:]:
            total_kernel = np.convolve(total_kernel, k)[:length]
        return total_kernel

    def get_corrections_kernel(self, kernel_list_before=None):
        kernel_list = [self.get_bounce_kernel(), self.get_skin_kernel(),
                       self.get_decay_kernel(), self.get_poly_kernel()]
        if kernel_list_before is not None:
            kernel_list_before.extend(kernel_list)
            return self.convolve_kernel(kernel_list_before,
                                        length=self.corrections_length())
        else:
            return self.convolve_kernel(kernel_list,
                                        length=self.corrections_length())

    def save_corrections_kernel(self, filename, kernel_list_before=None):
        if type(kernel_list_before) is not list:
            kernel_list_before = [kernel_list_before]
        save_kernel(self.get_corrections_kernel(kernel_list_before),
                    save_file=filename)
        return filename

