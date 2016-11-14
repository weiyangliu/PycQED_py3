import numpy as np
from pycqed.measurement import chevron_olli
from lmfit import Model
from scipy import special
# from plotting_tools import *

# heaviside function
heaviside = lambda t: np.double(t>=0)

# unit impulse function
square = lambda t, width=1: heaviside(t)-heaviside(t-width)

def fit_chevron(data, Vmin, Vmax, dV, dt, tmax, g, f_max, EC, asym, f_res, dict_guesses):
    fitting_model = chevron_model()
    v_vals = np.arange(Vmin, Vmax+0.5*dV, dV)
    t_vals = np.arange(0., tmax+0.5*dt, dt)
    fitting_model.set_param_hint('Vmin', value=Vmin, vary=False)
    fitting_model.set_param_hint('Vmax', value=Vmax, vary=False)
    fitting_model.set_param_hint('tmax', value=tmax, vary=False)
    fitting_model.set_param_hint('dV', value=dV, vary=False)
    fitting_model.set_param_hint('g', value=g, vary=False)
    fitting_model.set_param_hint('dt', value=dt, vary=False)
    fitting_model.set_param_hint('EC', value=EC, vary=False)
    fitting_model.set_param_hint('asym', value=asym, vary=False)
    fitting_model.set_param_hint('f_res', value=f_res, vary=False)
    fitting_model.set_param_hint('f_max', value=f_max, vary=False)
    fitting_model.set_param_hint('det0', value=f_max-f_res, vary=False)
    dict_params = {}
    for key in dict_guesses.keys():
        print(key, dict_guesses[key])
        fitting_model.set_param_hint(key, value=dict_guesses[key][0],
                                     min=dict_guesses[key][1],
                                     max=dict_guesses[key][2], vary=dict_guesses[key][3])
        dict_params.update({key: dict_guesses[key][0]})
    print(dict_params)
    params = fitting_model.make_params()
    curr_fit_res = fitting_model.fit(axis_dummy=np.concatenate((t_vals, v_vals)),
                                     data=data,
                                     params=params,
                                     method='powell')
    print(curr_fit_res.redchi)
    return curr_fit_res


def chevron_function(axis_dummy, det0, Vmin, Vmax, dV, g, tmax, dt,
                     a, v0, f_max, EC, asym, f_res,
                     amp_bounce, tau_bounce,
                     amp_decay, tau_decay,
                     alpha_skin):
    # f_func = lambda v: chevron_olli.qubit_freq_func(v, a, v0, f_max, EC, asym) - f_res
    f_func = lambda v: a*v+v0
    decay = lambda t, amp, tau: (1. - amp*np.exp(-t/tau))
    lowpass_s = lambda t, tau: heaviside(t+1)*(1-np.exp(-(t+1)/tau))
    bounce = lambda t, amp, tau: 1 + amp*(-1+np.double(t>tau))
    step_skineffect = lambda t, alpha: heaviside(t+1)*(1-special.erf(alpha/21./np.sqrt(t+1)))
    # print('no distortion')
    distortion = lambda t: bounce(t, amp_bounce, tau_bounce) * decay(t, amp_decay, tau_decay) * step_skineffect(t, alpha_skin)
    # print(Vmin,Vmax,dV,f_func,g,tmax,dt,distortion)
    # print('OMEGA PARAMS',a,v0,f_max,EC,asym,f_res)
    simulation = chevron_olli.chevron_voltage(det0=det0, Vmin=Vmin, Vmax=Vmax, dV=dV, omega=f_func,
                                             g=g, t=tmax, dt=dt, sf=distortion)
    result = np.array(simulation)
    # print('params', dt, tmax, Vmin, Vmax, dV)
    # print('result shape', result.shape)
    return result.flatten()


class chevron_model(Model):
    def __init__(self, *args, **kwargs):
        super(chevron_model, self).__init__(chevron_function, *args, **kwargs)

    def _residual(self, params, data, weights, **kwargs):
        "default residual:  (data-model)*weights"
        # print(len(self.eval(params, **kwargs)), len(data.flatten()))
        # print(params)
        diff = self.eval(params, **kwargs) - data.flatten()
        if weights is not None:
            diff *= weights
        return diff