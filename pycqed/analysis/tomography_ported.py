import qutip as qtp
import numpy as np
import time
import scipy
import os
from modules.analysis.tools.pytomo import pytomo as csdp_tomo
import uuid
import glob


class TomoAnalysis():

    """Performs state tomography based on an overcomplete set of measurements
     and calibration measurements. Uses qutip to calculate resulting basis states
     from applied rotations

    Uses binary counting as general guideline in ordering states. Calculates
     rotations by using the qutip library
    BEFORE YOU USE THIS SET THE CORRECT ORDER BY CHANGING
        'rotation_matrixes'
        'measurement_basis' + 'measurement_basis_labels'
        to values corresponding to your experiment
        and maybe 'readout_basis'
    """

    # The set of single qubit rotation matrixes used in the tomography
    # measurement (will be assumed to be used on all qubits)
    rotation_matrixes = [qtp.identity(2), qtp.sigmax(),
                         qtp.rotation(
                             qtp.sigmax(), np.pi / 2), qtp.rotation(qtp.sigmay(), np.pi / 2),
                         qtp.rotation(qtp.sigmax(), -np.pi / 2), qtp.rotation(qtp.sigmay(), -np.pi / 2)]
    measurement_operator_labels = ['I', 'X', 'x', 'y', '-x','-y']
    #MAKE SURE THE LABELS CORRESPOND TO THE ROTATION MATRIXES DEFINED ABOVE

    # The set of single qubit basis operators and labels
    measurement_basis = [
        qtp.identity(2), qtp.sigmaz(), qtp.sigmax(), qtp.sigmay()]
    measurement_basis_labels = ['I', 'Z', 'X', 'Y']
    # The operators used in the readout basis on each qubit
    readout_basis = [qtp.identity(2), qtp.sigmaz()]

    def __init__(self, measurements_cal, measurements_tomo, tomo_vars=None, n_qubits=2, n_quadratures=1, check_labels=True):
        """
        keyword arguments:
        measurements_cal --- Should be an array of length 2 ** n_qubits
        measurements_tomo --- Should be an array of length length(rotation_matrixes) ** n_qubits
        n_qubits --- default(2) the amount of qubits present in the expirement
        n_quadratures --- default(1(either I or Q)) The amount of complete measurement data sets. For example a combined IQ measurement has 2 measurement sets.
        tomo_vars  : since this tomo does not have access to the original data, the vars should be given by
                        tomo_var_i = 1 / N_i * np.var(M_i) where i stands for the data corresponding to rotation i. 

        """
        self.measurements_cal = measurements_cal
        self.measurements_tomo = measurements_tomo
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.n_quadratures = n_quadratures

        # Generate the vectors of matrixes that correspond to all measurements,
        # readout bases and rotations

        self.basis_vector = self._calculate_matrix_set(
            self.measurement_basis, n_qubits)
        self.readout_vector = self._calculate_matrix_set(
            self.readout_basis, n_qubits)
        self.rotation_vector = self._calculate_matrix_set(
            self.rotation_matrixes, n_qubits)

        self.tomo_vars = tomo_vars

        if check_labels is True:
            print(self.get_meas_operator_labels(n_qubits))
            print(self.get_basis_labels(n_qubits))



    def execute_pseudo_inverse_tomo(self, TE_correction_matrix=None):
        """
        Performs a linear tomography by simple inversion of the system of equations due to calibration points
        TE_correction_matrix: a matrix multiplying the calibration points to correct for estimated mixture due to Thermal excitation. 
        """

        # calculate beta positions in coefficient matrix
        coefficient_matrix = self._calculate_coefficient_matrix(TE_correction_matrix)
        basis_decomposition = np.zeros(4 ** self.n_qubits)
        # first skip beta0
        basis_decomposition[1:] = np.dot(
            np.linalg.pinv(coefficient_matrix[:, 1:]), self.measurements_tomo)
        # re-add beta0
        basis_decomposition[0] = 1
        # now recreate the rho
        rho = sum([basis_decomposition[i] * self.basis_vector[i] / (2 ** self.n_qubits)
                   for i in range(len(basis_decomposition))])
        return (basis_decomposition, rho)

    def execute_least_squares_physical_tomo(self, use_weights=False, show_time=True, ftol=0.01, xtol=0.001, full_output=0, max_iter=100,
                                            TE_correction_matrix = None):
        """
        Performs a least squares optimization using fmin_powell in order to get the closest physically realisable state.

        This is done by constructing a lower triangular matrix T consisting of 4 ** n qubits params
        Keyword arguments:
        use_weights : default(False) Weighs the quadrature data by the std in the estimator of the mean
                    : since this tomo does not have access to the original data, the vars should be given by
                        tomo_var_i = 1 / N_i * np.var(M_i) where i stands for the data corresponding to rotation i. 
        --- arguments for scipy fmin_powel method below, see the powel documentation
        """
        # first we calculate the measurement matrixes
        tstart = time.time()
        measurement_vector = []
        n_rot = len(self.rotation_matrixes) ** self.n_qubits
        # initiate with equal weights
        self.weights = np.ones(self.n_quadratures * n_rot)
        for quadrature in range(self.n_quadratures):
            betas = self._calibrate_betas(
                self.measurements_cal[quadrature * self.n_states: (1 + quadrature) * self.n_states],
                TE_correction_matrix)
            # determine the weights based on betas absulote difference and
            # accuracy
            if (use_weights):
                self.weights[
                    quadrature * n_rot:(1+quadrature) * n_rot] = 1 / self.tomo_vars[quadrature * n_rot:(1+quadrature) * n_rot]
            for rotation_index, rotation in enumerate(self.rotation_vector):
                measurement_vector.append(
                    betas[0] * rotation.dag() * self.readout_vector[0] * rotation)
                for i in range(1, len(betas)):
                    measurement_vector[n_rot * quadrature + rotation_index] += betas[
                        i] * rotation.dag() * self.readout_vector[i] * rotation
        # save it in the object for use in optimization
        self.measurement_vector = measurement_vector
        self.measurement_vector_numpy = [vec.full() for vec in measurement_vector]
        tlinear = time.time()
        # find out the starting rho by the linear tomo
        discard, rho0 = self.execute_pseudo_inverse_tomo(TE_correction_matrix)

        # now fetch the starting t_params from the cholesky decomp of rho
        tcholesky = time.time()
        T0 = np.linalg.cholesky(scipy.linalg.sqrtm((rho0.dag() * rho0).full()))
        t0 = np.zeros(4 ** self.n_qubits, dtype='complex' )
        di = np.diag_indices(2 ** self.n_qubits)
        tri = np.tril_indices(2 ** self.n_qubits, -1)
        t0[0:2 ** self.n_qubits] = T0[di]
        t0[2**self.n_qubits::2] = T0[tri].real
        t0[2**self.n_qubits+1::2] = T0[tri].imag
        topt = time.time()
        # minimize the likelihood function using scipy
        t_optimal = scipy.optimize.fmin_powell(
            self._max_likelihood_optimization_function, t0, maxiter=max_iter, full_output=full_output, ftol=ftol, xtol=xtol)
        if show_time is True:
            print(" Time to calc rotation matrixes %.2f " % (tlinear-tstart))
            print(" Time to do linear tomo %.2f " % (tcholesky-tlinear))
            print(" Time to build T %.2f " % (topt-tcholesky))
            print(" Time to optimize %.2f" % (time.time()-topt))
        return qtp.Qobj(self.build_rho_from_triangular_params(t_optimal),
         dims=[[2 for i in range(self.n_qubits)], [2 for i in range(self.n_qubits)]])



    def execute_SDPA_MC_2qubit_tomo(self,
                                    counts_tomo,
                                    counts_cal,
                                    N_total,
                                    used_bins = [0,2],
                                    n_runs = 100,
                                    array_like = False,
                                    correct_measurement_operators = True,
                                    TE_correction_matrix=None):
        """
        Executes the SDPDA tomo n_runs times with data distributed via a Multinomial distribution
        in order to get a list of rhos from which one can calculate errorbars on various derived quantities
        returns a list of Qobjects (the rhos).
        If array_like is set to true it will just return a 3D array of rhos
        """

        rhos= []
        for i in range(n_runs):
            #generate a data set based on multinomial distribution with means according to the measured data
            mc = [np.random.multinomial(sum(counts),(np.array(counts)+0.0) / sum(counts)) for counts in counts_tomo]
            rhos.append(self.execute_SDPA_2qubit_tomo(mc,
                                                      counts_cal,
                                                      N_total,
                                                      used_bins,
                                                      correct_measurement_operators,
                                                      TE_correction_matrix=TE_correction_matrix))

        if array_like:
            return np.array([rho.full() for rho in rhos])
        else:
            return rhos


    def execute_SDPA_2qubit_tomo(self, counts_tomo, counts_cal, N_total = 1, used_bins = [0,2],
                                 correct_measurement_operators=True, calc_chi_squared =False,
                                 correct_zero_count_bins=True, TE_correction_matrix = None):
        """
        Estimates a density matrix given single shot counts of 4 thresholded
        bins using a custom C semidefinite solver from Nathan Langford
        Each bin should correspond to a projection operator:
        0: 00, 1: 01, 2: 10, 3: 11
        The calibration counts are used in calculating corrections to the (ideal) measurement operators
        The tomo counts are used for the actual reconstruction.

        """
        if isinstance(used_bins, int):
            #allow for a single projection operator
            used_bins = [used_bins]

        Pm_corrected = self.get_meas_operators_from_cal(counts_cal,
                                                        correct_measurement_operators,
                                                        TE_correction_matrix)

        # If the tomography bins have zero counts, they not satisfy gaussian noise. If N>>1 then turning them into 1 fixes 
        # convergence problems without screwing the total statistics/estimate.
        if(np.sum(np.where(np.array(counts_tomo) == 0)) > 0):
                print("WARNING: Some bins contain zero counts, this violates gaussian assumptions. \n \
                        If correct_zero_count_bins=True these will be set to 1 to minimize errors")
        if correct_zero_count_bins:
            counts_tomo = [[int(b) if b > 0 else  1 for b in bin_counts] for bin_counts in counts_tomo]

        #Select the correct data based on the bins used
        #(and therefore based on the projection operators used)
        data = np.array([float(count[k]) for count in counts_tomo for k in used_bins] ).transpose()

        #get the total number of counts per tomo
        N = np.array([np.sum(counts_tomo, axis=1) for k in used_bins]).flatten()

        # add weights based on the total number of data points kept each run
        # N_total is a bit arbitrary but should be the average number of total counts of all runs, since in nathans code this
        # average is estimated as a parameter. 
        weights = N/float(N_total)

        #get the observables from the rotation operators and the bins kept(and their corresponding projection operators)
        measurement_operators = [Pm_corrected[k] for k in used_bins]
        observables = [(rot.dag() * measurement_operator * rot).full() for rot in self.rotation_vector for measurement_operator in measurement_operators]
        
        #calculate the density matrix using the csdp solver
        rho_nathan = csdp_tomo.tomo_state(data, observables, weights)
        n_estimate = rho_nathan.trace()
        rho = qtp.Qobj(rho_nathan / n_estimate,dims=[[2 for i in range(self.n_qubits)], [2 for i in range(self.n_qubits)]])
        
        if((np.abs(N_total - n_estimate) / N_total > 0.03)):
            print('WARNING estimated N(%d) is not close to provided N(%d) '% (n_estimate,N_total))

        if calc_chi_squared:
            chi_squared = self._state_tomo_goodness_of_fit(rho, data, N, observables)
            return rho, chi_squared
        else:
            return rho

    def get_meas_operators_from_cal(self, counts_cal, correct_measurement_operators=True, TE_correction_matrix=None):
        """
        Used in the thresholded tomography. Returns the set of corrected measurement operators
        """
        #setup the projection operators
        Pm_0 = qtp.projection(2,0,0)
        Pm_1 = qtp.projection(2,1,1)
        Pm_00 = qtp.tensor(Pm_0,Pm_0)
        Pm_11 = qtp.tensor(Pm_1,Pm_1)
        Pm_01 = qtp.tensor(Pm_0,Pm_1)
        Pm_10 = qtp.tensor(Pm_1,Pm_0)
        Pm = [Pm_00, Pm_01, Pm_10, Pm_11]
        #calculate bin probabilities normalized horizontally
        probs = (counts_cal / np.sum(counts_cal, axis = 1, dtype=float)[:,np.newaxis])
        if TE_correction_matrix is not None:
            probs = np.linalg.inv(TE_correction_matrix).dot(probs) 
        #print(probs)
        #correct the measurement operators based on calibration point counts
        if correct_measurement_operators is True:
            #just calc P_m_corrected = probs * P_m (matrix product)
            d = range(len(Pm))
            l = range(np.shape(counts_cal)[1])
            Pm_corrected = [sum(probs.T[i][j] * Pm[j] for j in d) for i in l]
        else:
            Pm_corrected = Pm
        # print 'Printing operators'
        # print Pm_corrected
        # print 'End of operators'
        return Pm_corrected


    def get_basis_labels(self, n_qubits):
        """
        Returns the basis labels in the same order as the basis vector is parsed.
        Requires self.measurement_basis_labels to be set with the correct order corresponding to the matrixes in self.measurement_basis
        """
        if(n_qubits > 1):
            return [x + y for x in self.get_basis_labels(n_qubits - 1)
                    for y in self.measurement_basis_labels]
        else:
            return self.measurement_basis_labels

    def get_meas_operator_labels(self, n_qubits):
        """
        Returns a vector of the rotations in order based on self.measurement_operator_labels
        """
        if(n_qubits > 1):
            return [x + y for x in self.get_meas_operator_labels(n_qubits - 1)
                    for y in self.measurement_operator_labels]
        else:
            return self.measurement_operator_labels

    def build_rho_from_triangular_params(self, t_params):
        # build the lower triangular matrix T
        T_mat = np.zeros(
            (2 ** self.n_qubits, 2 ** self.n_qubits), dtype="complex")
        di = np.diag_indices(2 ** self.n_qubits)
        T_mat[di] = t_params[0:2**self.n_qubits]
        tri = np.tril_indices(2 ** self.n_qubits, -1)
        T_mat[tri] = t_params[2**self.n_qubits::2]
        T_mat[tri] += 1j * t_params[2**self.n_qubits+1::2]
        rho = np.dot(np.conj(T_mat.T),  T_mat) / \
            np.trace(np.dot(np.conj(T_mat.T),  T_mat))
        return rho


##############################################################
#
#                    Private functions
#
##############################################################


    def _max_likelihood_optimization_function(self, t_params):
        """
        Optimization function that is evaluated many times in the maximum likelihood method.
        Calculates the difference between expected measurement values and the actual measurement values based on a guessed rho

        Keyword arguments:
        t_params : cholesky decomp parameters used to construct the initial rho
        Requires:
        self.weights :  weights per measurement vector used in calculating the loss
        """
        rho = self.build_rho_from_triangular_params(t_params)
        L = 0 + 0j
        for i in range(len(self.measurement_vector)):
            expectation = np.trace(
                np.dot(self.measurement_vector_numpy[i], rho))
            L += ((expectation -
                   self.measurements_tomo[i]) ** 2) * self.weights[i]
        return L

    def _calibrate_betas(self, measurements_cal, TE_correction_matrix=None):
        """calculates betas from calibration points for the initial measurement operator

        Betas are ordered by B0 -> II B1 -> IZ etc(binary counting)
        <0|Z|0> = 1, <1|Z|1> = -1

        Keyword arguments:
        measurements_cal --- array(2 ** n_qubits) should be ordered correctly (00, 01, 10, 11) for 2 qubits
        """
        cal_matrix = np.zeros((self.n_states, self.n_states))
        # get the coefficient matrix for the betas
        for i in range(self.n_states):
            for j in range(self.n_states):
                # perform bitwise AND and count the resulting 1s
                cal_matrix[i, j] = (-1)**(bin((i & j)).count("1"))
        #print("printing cal matrix",cal_matrix)
        # correct for thermal excitation via a matrix(can be obtained from the get_excited_calibration_points function)
        if TE_correction_matrix is not None:
            cal_matrix = TE_correction_matrix.dot(cal_matrix)
        # invert solve the simple system of equations
        betas = np.dot(np.linalg.inv(cal_matrix), measurements_cal)
        #print("betas: ", betas)
        return betas

    def _calculate_coefficient_matrix(self, TE_correction_matrix = None):
        """
        Calculates the coefficient matrix used when inversing the linear system of equations needed to find rho
        If there are multiple measurements present this will return a matrix of (n_quadratures * n_rotation_matrixes ** n_qubits) x n_basis_vectors
        """
        coefficient_matrix = np.zeros(
            (self.n_quadratures * len(self.rotation_matrixes) ** self.n_qubits, 4 ** self.n_qubits))
        n_rotations = len(self.rotation_matrixes) ** self.n_qubits
        # Now fill in 2 ** self.n_qubits betas into the coefficient matrix on
        # each row
        for quadrature in range(self.n_quadratures):
            # calibrate betas for this quadrature
            self.betas = self._calibrate_betas(
                self.measurements_cal[quadrature * self.n_states: (1 + quadrature) * self.n_states], TE_correction_matrix)
            for rotation_index in range(n_rotations):
                for beta_index in range(2 ** self.n_qubits):
                    (place, sign) = self._get_basis_index_from_rotation(
                        beta_index, rotation_index)
                    coefficient_matrix[
                        n_rotations * quadrature + rotation_index, place] = sign * self.betas[beta_index]
        return coefficient_matrix

    def _get_basis_index_from_rotation(self, beta_index, rotation_index):
        """Returns the position and sign of one of the betas in the coefficient matrix by checking
        to which basis matrix the readout matrix is mapped after rotation
        This is used in _calculate_coefficient_matrix
        """
        m = self.rotation_vector[rotation_index].dag(
        ) * self.readout_vector[beta_index] * self.rotation_vector[rotation_index]
        for basis_index, basis in enumerate(self.basis_vector):
            if(m == basis):
                return (basis_index, 1)
            elif(m == -basis):
                return (basis_index, -1)
        # if no basis is found raise an error
        raise Exception(
            'No basis vector found corresponding to the measurement rotation. Check that you have used Clifford Gates!')

    def _calculate_matrix_set(self, starting_set, n_qubits):
        """recursive function that returns len(starting_set) ** n_qubits
        measurement_basis states tensored with eachother based
        on the amount of qubits

        So for 2 qubits assuming your basis set is {I, X, Y, Z} you get II IX IY IZ XI XX XY XZ ...
        """
        if(n_qubits > 1):
            return [qtp.tensor(x, y) for x in self._calculate_matrix_set(starting_set, n_qubits - 1)
                    for y in starting_set]
        else:
            return starting_set


    #############################################################################################
    # CDSP tomo functions for likelihood.
    def _state_tomo_likelihood_function(self, rho, data, normalisations, observables, fixedweight=False):
        data_predicted = []
        for ii in range(len(data)):
            data_predicted.append((rho.full().dot(observables[ii])).trace()*normalisations[ii])

        data_predicted = np.array(data_predicted)

        if fixedweight:
            likely_function = np.sum( (data-data_predicted)**2/data )
        else:
            likely_function = np.sum( (data-data_predicted)**2/data_predicted )

        return likely_function

    """
        Calculates the goodness of fit. It has a normalisation which is just
        the sum of the counts in the superconducting case since there are no
        missing counts like in the photon case.
    """
    def _state_tomo_goodness_of_fit(self, rho, data, normalisations, observables,
                                   fixedweight=False, eig_cutoff=1e-6):

        likely_function = self._state_tomo_likelihood_function(rho, data, normalisations, observables,
                                                         fixedweight=fixedweight)
        num_data = len(data)
        num_eigs = np.sum(rho.eigenenergies()>eig_cutoff)
        rho_dim = rho.shape[0]
        num_dofs = num_eigs*(2*rho_dim-num_eigs)
        out = {}
        out['pure'] = likely_function / (num_data-(2*rho_dim-1))
        out['mixed'] = likely_function / (num_data-rho_dim**2)
        out['dofs'] = likely_function / (num_data-num_dofs)

        return out



#################################################################
#
# Data Generation (currently for 2 qubits only)
#
################################################################## 

comp_projectors = [qtp.ket2dm(qtp.tensor(qtp.basis(2,0), qtp.basis(2,0))),
                  qtp.ket2dm(qtp.tensor(qtp.basis(2,0), qtp.basis(2,1))),
                  qtp.ket2dm(qtp.tensor(qtp.basis(2,1), qtp.basis(2,0))),
                  qtp.ket2dm(qtp.tensor(qtp.basis(2,1), qtp.basis(2,1)))]

def generate_tomo_data(rho, M, R, N, M_bins = None):
    """
    Generates data for tomography. Both returns expectation values(used for average tomo) 
    or bin counts( if you use thresholded tomo). Generates a single multinomial set of counts to get both data types. 
    """

    #decompose the measurement operator in its spectrum
    eigenvals, eigenstates = M.eigenstates()
    if M_bins is None:
        M_bins =  comp_projectors
    # now decompose the 
    probs = []
    for state in eigenstates:
        #calculate probability of ending up in this state
        probs.append((R.dag() * qtp.ket2dm(state) * R * rho).tr().real)
    #run a multinomial distribution to determine the "experimental" measurement outcomes
    counts = np.random.multinomial(N, probs)
    # use the simulated percentages of states found to calc voltages
    expectations =  sum((counts / float(N)) * eigenvals)
    #calcultate bin counts via the projections of original eigenstates onto bin measurement operator. 
    bin_counts = [sum([counts[j] * (M_bins[i] * qtp.ket2dm(eigenstates[j])).tr().real
                  for j in range(len(eigenstates))])
                  for i in range(len(M_bins))]

    return bin_counts, expectations

def get_TE_correction_matrix(e_01, e_10, get_cal_points_instead=False):
    """
    Mixes the standard computational basis projectors to account for a certain thermal excitation fraction in qubit 1 and 2
    get_coefficient_matrix : return a matrix so one can correct the normal measurement operators used in TOMO
    If it is set to false, just return the mixed calibration points.
    """
    P = comp_projectors
    R = [   qtp.tensor(qtp.qeye(2), qtp.qeye(2)),
            qtp.tensor(qtp.qeye(2), qtp.sigmax()),
            qtp.tensor(qtp.sigmax(), qtp.qeye(2)),
            qtp.tensor(qtp.sigmax(), qtp.sigmax())]
    #calculate the effect TE on the 00 state using probabilities to be excited(or not)
    c_00 = (1-e_01) * (1-e_10) * P[0] + e_01 * (1-e_10) * P[1] + e_10 * (1-e_01) * P[2] + e_01 * e_10 * P[3]
    #find the other points via bit flip rotations
    c_01 = R[1] * c_00 * R[1].dag()
    c_10 = R[2] * c_00 * R[2].dag()
    c_11 = R[3] * c_00 * R[3].dag()

    if not get_cal_points_instead:
        return np.array([np.diag(c_00.full()), np.diag(c_01.full()), np.diag(c_10.full()), np.diag(c_11.full())]).T
    else:
        return [c_00, c_01, c_10, c_11]
