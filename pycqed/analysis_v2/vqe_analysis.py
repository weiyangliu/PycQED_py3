import lmfit
import numpy as np
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba


class VQE_optimization(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=False):
        super(self, __init__)(t_start, t_stop, options_dict,
                              extract_only, do_fitting)

        pdict = {'data': 'Data',
                 'R': 'ParHolder.distance_index'}
        nparams = ['R', 'data']
        self.params_dict = pdict
        self.numeric_params = nparams
