import numpy as np

# Module wide value definitions, can be overwritten after import
T1 = 18e-6
Tbus = 8e-6
tau_swap = 50e-9
tau_RO = 500e-9

valid_states = [0,  # no excitation
                1,  # excitation in qubit
                2]  # excitation in resonator


def swap(state):
    if state == 1:
        state = 2
    elif state == 2:
        state = 1
    else:
        state = 0
    return state


def decay(state):
    if state == 1:
        T_relax = T1
    else:
        T_relax = Tbus
    p_dec = 1-np.exp(-(tau_swap+tau_RO)/T_relax)
    if np.random.rand() < p_dec:
        state = 0
    return state


def pipulse(state):
    if state == 1:
        state = 0
    elif state == 0:
        state = 1
    return state


def measure(state, msmts):
    if state == 1:
        msmts.append(1)
    else:
        msmts.append(0)
    return state
