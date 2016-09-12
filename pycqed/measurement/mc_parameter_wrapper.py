'''
Module containing functions that wrap a QCodes parameter into a sweep or
detector function
'''
import qcodes as qc
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det
import time

def wrap_par_to_swf(parameter):
    '''
     - only soft sweep_functions
    '''
    sweep_function = swf.Sweep_function()
    sweep_function.sweep_control = 'soft'
    sweep_function.name = parameter.name
    sweep_function.parameter_name = parameter.label
    sweep_function.unit = parameter.units

    sweep_function.prepare = pass_function
    sweep_function.finish = pass_function
    sweep_function.set_parameter = parameter.set
    return sweep_function


def wrap_par_to_det(parameter, control='soft'):
    '''
    Todo:
     - only soft detector_functions
     - only single parameter
    '''
    detector_function = det.Detector_Function()
    detector_function.detector_control = control
    detector_function.name = parameter.name
    detector_function.value_names = [parameter.label]
    detector_function.value_units = [parameter.units]

    detector_function.prepare = pass_function
    detector_function.finish = pass_function
    detector_function.acquire_data_point = parameter.get
    detector_function.get_values = parameter.get
    return detector_function

def pass_function(**kw):
    pass


def wrap_func_to_det(func, name, value_names, units, control='soft',  **kw):
    detector_function = det.Detector_Function()
    detector_function.detector_control = control
    detector_function.name = name
    detector_function.value_names = value_names
    detector_function.value_units = units

    detector_function.prepare = pass_function
    detector_function.finish = pass_function
    def wrapped_func():
        return func(**kw)

    detector_function.acquire_data_point = wrapped_func
    detector_function.get_values = wrapped_func
    return detector_function

def wrap_par_remainder(par, remainder=1):
    new_par = qc.Parameter(name=par.name, label=par.label, units=par.units)
    def wrap_set(val):
        val = val % remainder
        par.set(val)
        par.get()
    new_par.set = wrap_set
    return new_par

def wrap_par_set_get(par):
    new_par = qc.Parameter(name=par.name, label=par.label, units=par.units)
    def wrap_set(val):
        par.set(val)
        par.get()
    new_par.set = wrap_set
    return new_par