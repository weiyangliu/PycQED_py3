# This snippet is based on example notebook 15 of PyGSTi 0.9.3

import os
import pycqed as pq
import pygsti
import pygsti.algorithms.germselection as germsel
import pygsti.algorithms.fiducialselection as fidsel
import pygsti.construction as constr

# Basic 1 qubit GST using the pi/2 rotations
gs_target = constr.build_gateset(
    [2],  # dimension of the Hilbert space
    [('Q0',)],  # This is a list of the target qubits
    ['Gi', 'Gx90', 'Gy90', 'Gx180', 'Gy180'],  # the labels of the gates
    ['I(Q0)', 'X(pi/2,Q0)', 'Y(pi/2,Q0)', 'X(pi, Q0)', 'Y(pi, Q0)'],
    prepLabels=['rho0'], prepExpressions=["0"],
    effectLabels=['E0'], effectExpressions=["1"],
    spamdefs={'plus': ('rho0', 'E0'),
              'minus': ('rho0', 'remainder')})

# The germ selection step can take quite some time if there are more gates.
# on my laptop the 3 primitives runs in a few seconds while the 5 primitives
# takes ~10 minutes
germs = germsel.generate_germs(gs_target)
prepFiducials, measFiducials = fidsel.generate_fiducials(gs_target)
# As generating the germs takes some time we want to store them.
fp = os.path.join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                  '_pygsti_Gatesequences', '1Q_5prim_germs.txt')
pygsti.io.write_gatestring_list(fp, germs, "1Q 5 primitives germs")
# They can be read back using
# germs = pygsti.io.load_gatestring_list(fp)


maxLengths = [0] + [2**n for n in range(10)]  # goes up to 512
listOfExperiments = constr.make_lsgst_experiment_list(gs_target.gates.keys(), prepFiducials,
                                                      measFiducials, germs, maxLengths)
L = maxLengths[-1]
N = len(listOfExperiments)

# At this point there is a set of experiments to perform.
fp = os.path.join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                  '_pygsti_Gatesequences', '1Q_GST_5prim_L{}_N{}.txt'.format(L, N))
pygsti.io.write_empty_dataset(fp, listOfExperiments)

# There are two distinct considerations that force us to split up
# the experiment into multiple segments.
#  - The data acquisition:
#           - per 8000 individual shots
#           - 256 averaged segments
#  - The instruction limit


##############################################################################
######################### And the 2 qubit example ############################
##############################################################################

# This snippet is based on example notebook 15 of PyGSTi 0.9.3

# Basic 1 qubit GST using the pi/2 rotations
gs_target = constr.build_gateset(
    [4],  # dimension of the Hilbert space
    [('Q0', 'Q1')],  # This is a list of the target qubits
    # the labels of the gates
    ['Gii', 'Gix', 'Giy', 'Gxi', 'Gyi', 'Gxx', 'Gyy'],
    ['I(Q1):I(Q0)',
     'I(Q1):X(pi/2,Q0)', 'I(Q1):Y(pi/2,Q0)',
     'X(pi/2, Q1):I(Q0)', 'X(pi/2, Q1):I(Q0)',
     'X(pi/2, Q1):X(pi, Q0)', 'X(pi/2, Q1):Y(pi, Q0)'],
    prepLabels=['rho0'], prepExpressions=["0"],
    effectLabels=['E0'], effectExpressions=["1"],
    spamdefs={'plus': ('rho0', 'E0'),
              'minus': ('rho0', 'remainder')})

# The germ selection step can take quite some time if there are more gates.
# on my laptop the 3 primitives runs in a few seconds while the 5 primitives
# takes ~10 minutes
germs = germsel.generate_germs(gs_target)
prepFiducials, measFiducials = fidsel.generate_fiducials(gs_target)
# As generating the germs takes some time we want to store them.
fp = os.path.join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                  '_pygsti_Gatesequences', '2Q_cardinal_germs.txt')
pygsti.io.write_gatestring_list(fp, germs, "2Q cardinal germs")
# They can be read back using
# germs = pygsti.io.load_gatestring_list(fp)


maxLengths = [0] + [2**n for n in range(10)]  # goes up to 512
listOfExperiments = constr.make_lsgst_experiment_list(gs_target.gates.keys(), prepFiducials,
                                                      measFiducials, germs, maxLengths)
L = maxLengths[-1]
N = len(listOfExperiments)

# At this point there is a set of experiments to perform.
fp = os.path.join(pq.__path__[0], 'measurement', 'gate_set_tomography',
                  '_pygsti_Gatesequences', '2Q_cardinal_GST_L{}_N{}.txt'.format(L, N))
pygsti.io.write_empty_dataset(fp, listOfExperiments)

# There are two distinct considerations that force us to split up
# the experiment into multiple segments.
#  - The data acquisition:
#           - per 8000 individual shots
#           - 256 averaged segments
#  - The instruction limit is 32k (about a factor 10 larger than the number
#    of gatestrings). I expect to have to split the experment up in 2 for this
#    reason as the loop instruction, the readouts and the initializing weights
#    also take up instructions.
