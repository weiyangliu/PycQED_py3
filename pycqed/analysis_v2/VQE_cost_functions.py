import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/repository/PycQED_py3')


class VQE_cost_functions(object):

    def __init__(self,
                 path_file,
                 expect_values):
        """
        Class is instantiated with the hydrogen data path file and measured expectation values.
        The expectation values are in the following order II, IZ, ZI, ZZ, XX, YY

        """
        self.path_file = path_file
        self.hydrogen_data = np.loadtxt(self.path_file, unpack=False)
        self.interatomic_distances = self.hydrogen_data[:, 0]
        self.weight_of_pauli_terms = self.hydrogen_data[:, 1:7]
        self.expect_values = expect_values

    def cost_function_bare_VQE(self, distance_index):
        cost_func_bare = np.dot(
            self.expect_values, self.weight_of_pauli_terms[distance_index, :])
        return cost_func_bare
