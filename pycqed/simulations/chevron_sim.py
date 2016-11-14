"""
Based on Olli's mathematica notebook used to simulate chevrons
"""

import numpy as np
from scipy.linalg import expm

ham = lambda e, g: np.array([[0.5*e, g], [g, -0.5*e]])
evol = lambda e, g, dt: expm(dt*1j*ham(e, g))


def rabisim(efun, g, t, dt):
    """
    This function returns the evolution of a system described by the hamiltonian:
            H = efun sigma_z + g sigma_x
    Inputs:
            efun,   Function that returns the energy parameter vs time.s
            g,      Coupling parameter
            t,      Final time of the evolution
            dt,     Stepsize of the time evolution
    Outputs:
            f_vec,  Evolution for times (1, 1+dt, ..., t)
    """
    s0 = np.array([1, 0])
    ts = np.arange(1., np.round(t)+0.5*np.round(dt), np.round(dt))
    # print(len(ts),t,dt)
    f = lambda st, ti: np.dot(evol(efun(ti), g, dt), st)
    f_vec = np.zeros((len(ts), 2), dtype=np.complex128)
    f_vec[0, :] = s0
    for i, t in enumerate(ts[:-1]):
        f_vec[i+1, :] = f(f_vec[i], t)
    return f_vec

qamp = lambda vec: np.abs(vec[:, 1])**2


def chevron(e0, emin, emax, n, g, t, dt, sf):
    energy_func = lambda energy, t: e0*(1.-(energy*sf(t))**2)
    energy_vec = np.arange(1+emin, 1+emax, (emax-emin)/(n-1))
    chevron_vec = []
    for ee in energy_vec:
        print(ee,energy_func(ee, t), g, t, dt)
        chevron_vec.append(
            qamp(rabisim(lambda t: energy_func(ee, t), g, t, dt)))
    return np.array(chevron_vec)

def chevron_voltage(Vmin, Vmax, dV, omega, g, t, dt, sf):
    # print(Vmin,Vmax,dV,omega,g,t,dt,sf)
    voltage_vec = np.arange(1+Vmin, 1+Vmax+dV*0.5, dV)
    energy_func = lambda v, t: (1.-(omega(v)*sf(t))**2)
    chevron_vec = []
    for vv in voltage_vec:
        # print(vv,energy_func(vv-1., t), g, t, dt)
        chevron_vec.append(
            qamp(rabisim(lambda t: energy_func(vv, t), g, t, dt)))
    return np.array(chevron_vec)

qubit_freq_func = lambda v, a, v0, f_max, EC, asym: (f_max + EC) * (asym**2 + (1-asym**2)*np.cos(a*(v-v0))**2)**0.25 - EC