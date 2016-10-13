import numpy as np
from scipy import special


def bounce_kernel(amp=0.02, time=4, length=601):
    """
    Generates a bounce kernel, with the specified parameters.

    kernel_step_function:
        heaviside(t) + amp*heaviside(t-time)
    """
    t_kernel = np.arange(length)
    bounce_pairs = [[amp, time]]
    # kernel_bounce = kernel_generic(htilde_bounce, t_kernel, bounce_pairs)
    # kernel_bounce /= np.sum(kernel_bounce)
    bounce_kernel_step = step_bounce(t_kernel, bounce_pairs)
    kernel_bounce = np.zeros(bounce_kernel_step.shape)
    kernel_bounce[0] = bounce_kernel_step[0]
    kernel_bounce[1:] = bounce_kernel_step[1:]-bounce_kernel_step[:-1]
    kernel_bounce /= np.sum(kernel_bounce)

    return kernel_bounce


def decay_kernel(amp=1., tau=11000, length=20000):
    """
    Generates a decay kernel, with the specified parameters

    kernel_step_function
        1 + amp*np.exp(-t_kernel/tau)
    """
    t_kernel = np.arange(length)
    decay_kernel_step = 1 + amp*np.exp(-t_kernel/tau)
    decay_kernel = np.zeros(decay_kernel_step.shape)
    decay_kernel[0] = decay_kernel_step[0]
    decay_kernel[1:] = decay_kernel_step[1:]-decay_kernel_step[:-1]
    return decay_kernel


def skin_kernel(alpha=0., length=601):
    """
    Generates a skin effect kernel, with the specified parameters

    kernel_step_function
        heaviside(t+1)*(1-errf(alpha/21./np.sqrt(t+1)))
    """
    t_kernel = np.arange(length)
    # kernel_skineffect = kernel_generic(htilde_skineffect,t_kernel,alpha)
    # kernel_skineffect /= np.sum(kernel_skineffect)
    skineffect_kernel_step = step_skineffect(t_kernel, alpha)
    kernel_skineffect = np.zeros(skineffect_kernel_step.shape)
    kernel_skineffect[0] = skineffect_kernel_step[0]
    kernel_skineffect[1:] = skineffect_kernel_step[
        1:]-skineffect_kernel_step[:-1]
    kernel_skineffect /= np.sum(kernel_skineffect)
    return kernel_skineffect


def poly_kernel(a=0, b=11000, c=0, length=30000):
    """
    Generates a polynomial kernel(like the one used for bias-tee),
    with the specified parameters
    """
    t_kernel = np.arange(length)
    poly_kernel_step = a*t_kernel**2+b*t_kernel+c
    kernel_poly = np.zeros(poly_kernel_step.shape)
    kernel_poly[0] = poly_kernel_step[0]
    kernel_poly[1:] = poly_kernel_step[1:]-poly_kernel_step[:-1]
    return kernel_poly


# response functions for the low-pass filter arising from the skin effect of a coax
# model parameter: alpha (attenuation at 1 GHz)
step_skineffect = lambda t, alpha: heaviside(
    t+1)*(1-special.erf(alpha/21./np.sqrt(t+1)))
htilde_skineffect = lambda t, alpha: htilde(step_skineffect, t, alpha)


# heaviside function
heaviside = lambda t: t >= 0

# unit impulse function
square = lambda t, width=1: heaviside(t)-heaviside(t-width)

# generic function calculating the htilde (discrete impulse response) from
# a known step function
htilde = lambda fun, t, params, width=1: fun(t, params)-fun(t-width, params)

filter_matrix_generic = lambda fun, t, *params: np.sum(np.array(map(lambda ii:
                                                                    np.diag(fun(t[ii], *params)*np.ones(len(t)-ii), k=-ii), range(len(t)))), 0)
kernel_generic = lambda fun, t, * \
    params: np.linalg.inv(filter_matrix_generic(fun, t, *params))[:, 0]

step_bounce = lambda t, pairs: heaviside(
    t) + np.sum(np.array([pair[0]*heaviside(t - pair[1]) for pair in pairs]), 0)
htilde_bounce = lambda t, pairs: htilde(step_bounce, t, pairs)
