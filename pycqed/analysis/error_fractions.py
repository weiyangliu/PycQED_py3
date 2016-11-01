import numpy as np


def conventional_error_fraction(measurements, target_operation=0):
    """
    Calculates conventional error fraction as defined in restless paper
    input:
        measurements (list, array)  measurement outcomes (0, 1)
        target_operation (int):     0 (Identity) or 1 (bit-flip)
    output:
        ec = sum_n=1^N m_n          # if target_operation is 0
    """
    if target_operation == 0:
        ec = np.mean(measurements)
    elif target_operation == 1:
        ec = 1-np.mean(measurements)
    else:
        raise ValueError('target operation "{}"'.format(target_operation)
                         + ' not supported should be (0,1)')
    return ec


def restless_error_fraction(measurements, target_operation=0):
    """
    Calculates restless error fraction as defined in restless paper
    input:
        measurements (list, array)  measurement outcomes (0, 1)
        target_operation (int):     0 (Identity) or 1 (bit-flip)
    output:
        er = sum_{n=2}^N m_n == m_{n-1}    # equal if target_operation is 1
    """
    # convert to array to allow easy binary derivative
    measurements = np.array(measurements)
    binary_der = abs(measurements[1:] - measurements[:-1])
    if target_operation == 0:
        er = np.mean(binary_der)
    elif target_operation == 1:
        er = 1-np.mean(binary_der)
    else:
        raise ValueError('target operation "{}"'.format(target_operation)
                         + ' not supported should be (0,1)')
    return er
