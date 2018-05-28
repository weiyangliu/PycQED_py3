import numpy as np
import sys
sys.path.append('D:/repository/PycQED_py3')
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import composite_analysis as RA
from pycqed.analysis import measurement_analysis as ma


class VQE_analysis(object):

    def __init__(self,
                 path_file,
                 timestamp_start,
                 timestamp_stop,
                 T1_signalling = False,
                 scan_label ='VQE'):
        """
        Class is instantiated with the hydrogen data path file and measured expectation values.
        The expectation values are in the following order II, IZ, ZI, ZZ, XX, YY

        """
        self.path_file = path_file
        self.hydrogen_data = np.loadtxt(self.path_file, unpack=False)
        self.interatomic_distances = self.hydrogen_data[:, 0]
        self.weight_of_pauli_terms = self.hydrogen_data[:, 1:7]
        ##look for tomography results in the data file
        self.t_list = a_tools.get_timestamps_in_range(timestamp_start=timestamp_start,
                                                      timestamp_end=timestamp_stop,
                                                      label=scan_label)
        pdict = {'theta':'ParHolder.theta',
                 'phi':'ParHolder.phi',
                 'II':'Analysis.tomography_results.II',
                 'IZ':'Analysis.tomography_results.IZ',
                 'ZI':'Analysis.tomography_results.ZI',
                 'ZZ':'Analysis.tomography_results.ZZ',
                 'XX':'Analysis.tomography_results.XX',
                 'YY':'Analysis.tomography_results.YY'}
        opt_dict = {'scan_label': scan_label,
                    'exact_label_match': True}
        nparams = ['theta', 'phi', 'II', 'IZ', 'ZI', 'ZZ', 'XX', 'YY']
        scans_VQE = RA.quick_analysis(t_start=self.t_list[0],
                                      t_stop=self.t_list[-1],
                                      options_dict=opt_dict,
                                      params_dict_TD=pdict,
                                      numeric_params=nparams)
        theta_array = scans_VQE.TD_dict['theta']
        phi_array = scans_VQE.TD_dict['phi']
        self.II_array = scans_VQE.TD_dict['II']
        self.IZ_array = scans_VQE.TD_dict['IZ']
        self.ZI_array = scans_VQE.TD_dict['ZI']
        self.ZZ_array = scans_VQE.TD_dict['ZZ']
        self.XX_array = scans_VQE.TD_dict['XX']
        # print(self.XX_array)
        self.YY_array = scans_VQE.TD_dict['YY']
        # print(self.YY_array)
        self.phi_vec, self.theta_vec = np.unique(phi_array), np.unique(theta_array)
        # print('Printing the length of theta_vec and phi_vec')
        # print(self.phi_vec.shape,self.theta_vec.shape)
        self.data_array = np.column_stack((theta_array, phi_array,
                                           self.II_array, self.IZ_array,
                                           self.ZI_array, self.ZZ_array,
                                           self.XX_array, self.YY_array))
        # print('Printing the dims of data_array')
        # print(self.data_array.shape)

        #save the data_array in a .npy file
        #structured as |theta| phi| II| IZ| ZI| XX| YY|
        np.save('theta_phi_expect_values_{}_{}'.format(self.t_list[0],self.t_list[-1]), self.data_array)
        # self.T1_signalling = T1_signalling

    def cost_function_bare_VQE(self, distance_index, expect_values_theta_phi):
        cost_func_bare = np.dot(expect_values_theta_phi, self.weight_of_pauli_terms[distance_index, :])
        return cost_func_bare

    def get_energy_landscape(self, distance_index):
        """
        Computes the energy landscape for a specified distance index.
        Accesses the data_array attribute and multiplies every row of this array (which correspond to the 6 expectation values)
        with the g prefacors of a specified distance index.
        """
        energy_for_theta_phi = np.zeros(len(self.data_array[:,0]))
        for i in np.arange(len(self.data_array[:,0])):
            energy_for_theta_phi[i] = self.cost_function_bare_VQE(distance_index,
                                                                  self.data_array[i, 2:])
        energy_landscape = energy_for_theta_phi.reshape((len(self.phi_vec),
                                                         len(self.theta_vec)))
        return energy_landscape

    def find_min_landscape(self, distance_index):
        """
        gets energy landscpe, finds it's minimum and
        then returns phi_min and theta_min
        """
        energy_landscape = self.get_energy_landscape(distance_index)
        print('######################################')
        print('Printing the shape of Energy Landscape')
        print(energy_landscape.shape)
        min_idx = np.unravel_index(np.argmin(energy_landscape, axis=None),energy_landscape.shape)
        print('######################################')
        print('Printing min_idx')
        print(min_idx)
        energy_min = energy_landscape[min_idx]
        theta_min = self.theta_vec[min_idx[1]]
        phi_min = self.phi_vec[min_idx[0]]

        min_stack = np.array([energy_min,
                              theta_min,
                              phi_min,
                              self.II_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.IZ_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.ZI_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.ZZ_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.XX_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.YY_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx]])


        # if self.T1_signalling:
        #     min_stack_new = self.do_T1_signalling(min_stack)

        return min_stack

    # def do_T1_signalling(self, min_stack):
    #     II_experimental = min_stack[3]
    #     IZ_experimental = min_stack[4]
    #     ZI_experimental = min_stack[5]
    #     ZZ_experimental = min_stack[6]
    #     XX_experimental = min_stack[7]
    #     YY_experimental = min_stack[8]
    #     II_mitigated = (II_experimental - ZZ_experimental)/(1 - ZZ_experimental)
    #     IZ_mitigated = (IZ_experimental - ZI_experimental)/(1 - ZZ_experimental)
    #     ZI_mitigated = (ZI_experimental - IZ_experimental)/(1 - ZZ_experimental)
    #     ZZ_mitigated = (ZZ_experimental - II_experimental)/(1 - ZZ_experimental)
    #     XX_mitigated = (XX_experimental + YY_experimental)/(1 - ZZ_experimental)
    #     YY_mitigated = (XX_experimental + YY_experimental)/(1 - ZZ_experimental)
    #     min_stack = np.array([min_stack[0],
    #                           min_stack[1],
    #                           min_stack[2],
    #                           II_mitigated,
    #                           IZ_mitigated,
    #                           ZI_mitigated,
    #                           ZZ_mitigated,
    #                           XX_mitigated,
    #                           YY_mitigated])
    #     return min_stack


    def compute_dissociation_curve(self):
        """
        Uses find_min_landscape as a subroutine to find values of useful quantities for which
        energy is minimized.Does this for every interatomic distance.

        """
        min_stack_global = np.zeros((len(self.interatomic_distances),9))
        for i in np.arange(len(self.interatomic_distances)):
            min_stack_global[i,:] = self.find_min_landscape(i)
        minimized_data = min_stack_global
        np.save('data_after_minimization_{}_{}'.format(self.t_list[0], self.t_list[-1]), minimized_data)
        return minimized_data

    def get_pauli_ops(self):
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.identity(2)
        II = np.kron(I, I)
        IZ = np.kron(I, Z)
        ZI = np.kron(Z, I)
        ZZ = np.kron(Z, Z)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        return II, IZ, ZI, ZZ, XX, YY

    def get_hamiltonian(self, distance_index):
        terms = self.get_pauli_ops()
        gs = self.weight_of_pauli_terms[distance_index, :]
        ham = np.zeros((4,4), dtype=np.complex128)
        for i,g in enumerate(gs):
            ham += g*terms[i]
        return ham


class VQE_analysis_phase_corr(object):

    def __init__(self,
                 path_file,
                 timestamp_start,
                 timestamp_stop,
                 T1_signalling = False,
                 scan_label ='VQE'):
        """
        Class is instantiated with the hydrogen data path file and measured expectation values.
        The expectation values are in the following order II, IZ, ZI, ZZ, XX, YY

        """
        self.path_file = path_file
        self.hydrogen_data = np.loadtxt(self.path_file, unpack=False)
        self.interatomic_distances = self.hydrogen_data[:, 0]
        self.weight_of_pauli_terms = self.hydrogen_data[:, 1:7]
        ##look for tomography results in the data file
        self.t_list = a_tools.get_timestamps_in_range(timestamp_start=timestamp_start,
                                                      timestamp_end=timestamp_stop,
                                                      label=scan_label)
        pdict = {'theta':'ParHolder.theta',
                 'phi':'ParHolder.phi',
                 'II':'Analysis.tomography_results.II',
                 'IZ':'Analysis.tomography_results.IZ',
                 'IX':'Analysis.tomography_results.IX',
                 'IY':'Analysis.tomography_results.IY',
                 'ZI':'Analysis.tomography_results.ZI',
                 'ZZ':'Analysis.tomography_results.ZZ',
                 'ZX':'Analysis.tomography_results.ZX',
                 'ZY':'Analysis.tomography_results.ZY',
                 'XI':'Analysis.tomography_results.XI',
                 'XZ':'Analysis.tomography_results.XZ',
                 'XX':'Analysis.tomography_results.XX',
                 'XY':'Analysis.tomography_results.XY',
                 'YI':'Analysis.tomography_results.YI',
                 'YZ':'Analysis.tomography_results.YZ',
                 'YX':'Analysis.tomography_results.YX',
                 'YY':'Analysis.tomography_results.YY'}
        opt_dict = {'scan_label': scan_label,
                    'exact_label_match': True}
        nparams = ['theta', 'phi', 'II', 'IZ', 'IX', 'IY',
                   'ZI', 'ZZ', 'ZX', 'ZY', 'XI', 'XZ', 'XX',
                   'XY', 'YI', 'YZ', 'YX', 'YY']
        scans_VQE = RA.quick_analysis(t_start=self.t_list[0],
                                      t_stop=self.t_list[-1],
                                      options_dict=opt_dict,
                                      params_dict_TD=pdict,
                                      numeric_params=nparams)
        theta_array = scans_VQE.TD_dict['theta']
        phi_array = scans_VQE.TD_dict['phi']

        self.II_array = []
        self.IZ_array = []
        self.ZI_array = []
        self.ZZ_array = []
        self.XX_array = []
        self.YY_array = []

        for i,this_phi in enumerate(phi_array):
            exp_values = [scans_VQE.TD_dict[key][i] for key in ['II', 'IZ', 'IX', 'IY','ZI', 'ZZ', 'ZX', 'ZY', 'XI', 'XZ', 'XX','XY', 'YI', 'YZ', 'YX', 'YY']]
            exp_values = self.correct_phase(exp_values,this_phi*np.pi/180.)
            self.II_array.append(exp_values[0])
            self.IZ_array.append(exp_values[1])
            self.ZI_array.append(exp_values[4])
            self.ZZ_array.append(exp_values[5])
            self.XX_array.append(exp_values[10])
            self.YY_array.append(exp_values[15])

        self.II_array = np.array(self.II_array)
        self.IZ_array = np.array(self.IZ_array)
        self.ZI_array = np.array(self.ZI_array)
        self.ZZ_array = np.array(self.ZZ_array)
        self.XX_array = np.array(self.XX_array)
        self.YY_array = np.array(self.YY_array)
        # print(self.YY_array)
        self.phi_vec, self.theta_vec = np.unique(phi_array), np.unique(theta_array)
        # print('Printing the length of theta_vec and phi_vec')
        # print(self.phi_vec.shape,self.theta_vec.shape)
        self.data_array = np.column_stack((theta_array, phi_array,
                                           self.II_array, self.IZ_array,
                                           self.ZI_array, self.ZZ_array,
                                           self.XX_array, self.YY_array))
        # print('Printing the dims of data_array')
        # print(self.data_array.shape)

        #save the data_array in a .npy file
        #structured as |theta| phi| II| IZ| ZI| XX| YY|
        np.save('theta_phi_expect_values_{}_{}'.format(self.t_list[0],self.t_list[-1]), self.data_array)
        # self.T1_signalling = T1_signalling


    def correct_phase(self, exp_values,desired_angle):
        # constructs operators
        op_list = [ma.qtp.qeye(2),ma.qtp.sigmaz(),ma.qtp.sigmax(),ma.qtp.sigmay()]
        basis_operators = np.zeros((16,4,4),dtype=np.complex)
        for i in range(16):
            idx_0 = i % 4
            idx_1 = ((i - idx_0) // 4) % 4
            basis_operators[i,:,:] = ma.qtp.tensor(op_list[idx_1],op_list[idx_0]).full()
        exp_values =np.array(exp_values)
        # reconstructs rho
        rho = np.zeros((4,4),dtype=np.complex)
        for i in range(16):
            rho += exp_values[i]*basis_operators[i,:,:]
        # get the phase
        angle_0110 = np.angle(rho[1,2])
        rotation_angle = desired_angle-angle_0110
        # corrects for it
        rot_zphi = ma.qtp.rotation(ma.qtp.tensor(ma.qtp.sigmaz(),
                                                 ma.qtp.qeye(2)),
                                   rotation_angle)
        rho = (rot_zphi.conj())*ma.qtp.Qobj(rho,dims=[[2,2],[2,2]])*rot_zphi
        rho = 0.25*rho

        exp_values_corr = np.zeros(16)
        for i in range(16):
            idx_0 = i % 4
            idx_1 = ((i - idx_0) // 4) % 4
            exp_values_corr[i] = np.real_if_close((rho*ma.qtp.tensor(op_list[idx_1],op_list[idx_0])).tr())
        return exp_values_corr


    def cost_function_bare_VQE(self, distance_index, expect_values_theta_phi):
        cost_func_bare = np.dot(expect_values_theta_phi, self.weight_of_pauli_terms[distance_index, :])
        return cost_func_bare

    def get_energy_landscape(self, distance_index):
        """
        Computes the energy landscape for a specified distance index.
        Accesses the data_array attribute and multiplies every row of this array (which correspond to the 6 expectation values)
        with the g prefacors of a specified distance index.
        """
        energy_for_theta_phi = np.zeros(len(self.data_array[:,0]))
        for i in np.arange(len(self.data_array[:,0])):
            energy_for_theta_phi[i] = self.cost_function_bare_VQE(distance_index,
                                                                  self.data_array[i, 2:])
        energy_landscape = energy_for_theta_phi.reshape((len(self.phi_vec),
                                                         len(self.theta_vec)))
        return energy_landscape

    def find_min_landscape(self, distance_index):
        """
        gets energy landscpe, finds it's minimum and
        then returns phi_min and theta_min
        """
        energy_landscape = self.get_energy_landscape(distance_index)
        print('######################################')
        print('Printing the shape of Energy Landscape')
        print(energy_landscape.shape)
        min_idx = np.unravel_index(np.argmin(energy_landscape, axis=None),energy_landscape.shape)
        print('######################################')
        print('Printing min_idx')
        print(min_idx)
        energy_min = energy_landscape[min_idx]
        theta_min = self.theta_vec[min_idx[1]]
        phi_min = self.phi_vec[min_idx[0]]

        min_stack = np.array([energy_min,
                              theta_min,
                              phi_min,
                              self.II_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.IZ_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.ZI_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.ZZ_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.XX_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx],
                              self.YY_array.reshape((len(self.phi_vec),len(self.theta_vec)))[min_idx]])


        # if self.T1_signalling:
        #     min_stack_new = self.do_T1_signalling(min_stack)

        return min_stack

    # def do_T1_signalling(self, min_stack):
    #     II_experimental = min_stack[3]
    #     IZ_experimental = min_stack[4]
    #     ZI_experimental = min_stack[5]
    #     ZZ_experimental = min_stack[6]
    #     XX_experimental = min_stack[7]
    #     YY_experimental = min_stack[8]
    #     II_mitigated = (II_experimental - ZZ_experimental)/(1 - ZZ_experimental)
    #     IZ_mitigated = (IZ_experimental - ZI_experimental)/(1 - ZZ_experimental)
    #     ZI_mitigated = (ZI_experimental - IZ_experimental)/(1 - ZZ_experimental)
    #     ZZ_mitigated = (ZZ_experimental - II_experimental)/(1 - ZZ_experimental)
    #     XX_mitigated = (XX_experimental + YY_experimental)/(1 - ZZ_experimental)
    #     YY_mitigated = (XX_experimental + YY_experimental)/(1 - ZZ_experimental)
    #     min_stack = np.array([min_stack[0],
    #                           min_stack[1],
    #                           min_stack[2],
    #                           II_mitigated,
    #                           IZ_mitigated,
    #                           ZI_mitigated,
    #                           ZZ_mitigated,
    #                           XX_mitigated,
    #                           YY_mitigated])
    #     return min_stack


    def compute_dissociation_curve(self):
        """
        Uses find_min_landscape as a subroutine to find values of useful quantities for which
        energy is minimized.Does this for every interatomic distance.

        """
        min_stack_global = np.zeros((len(self.interatomic_distances),9))
        for i in np.arange(len(self.interatomic_distances)):
            min_stack_global[i,:] = self.find_min_landscape(i)
        minimized_data = min_stack_global
        np.save('data_after_minimization_{}_{}'.format(self.t_list[0], self.t_list[-1]), minimized_data)
        return minimized_data

    def get_pauli_ops(self):
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.identity(2)
        II = np.kron(I, I)
        IZ = np.kron(I, Z)
        ZI = np.kron(Z, I)
        ZZ = np.kron(Z, Z)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        return II, IZ, ZI, ZZ, XX, YY

    def get_hamiltonian(self, distance_index):
        terms = self.get_pauli_ops()
        gs = self.weight_of_pauli_terms[distance_index, :]
        ham = np.zeros((4,4), dtype=np.complex128)
        for i,g in enumerate(gs):
            ham += g*terms[i]
        return ham

