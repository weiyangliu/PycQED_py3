import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
# import dataprep for tomography module
# import tomography module
# using the data prep module of analysis V2 
from pycqed.analysis_v2 import tomography_dataprep as dataprep
from pycqed.analysis import measurement_analysis as ma 
try:
	import qutip as qt
except ImportError as e:
	logging.warning('Could not import qutip, tomo code will not work')

    

class ExpectationValueCalculation(ma.MeasurementAnalysis):

    def __init__(self, auto=True, label='', timestamp=None,
                 MLE=False, target_cardinal=None, target_bell=None,
                 start_shot=0, end_shot=-1,
                 verbose=0,
                 single_shots=True,
                 fig_format='png',
                 q0_label='q0',
                 q1_label='q1', close_fig=True, **kw):
        self.label = label
        self.timestamp = timestamp
        self.target_cardinal = target_cardinal
        self.target_bell = target_bell
        self.start_shot = start_shot
        self.end_shot = end_shot
		self.fig_format = fig_format
        self.q0_label = q0_label
        self.q1_label = q1_label
        self.n_states = 2 ** 2
        

    
    self.get_naming_and_values()
    # hard coded number of segments for a 2 qubit state tomography
    # constraint imposed by UHFLI
    self.nr_segments = 16
    # self.exp_name = os.path.split(self.folder)[-1][7:]

    avg_h1 = self.measured_values[0]
    avg_h2 = self.measured_values[1]
    avg_h12 = self.measured_values[2]

    # Binning all the points required for the tomo
    h1_00 = np.mean(avg_h1[36:36+7])
    h1_01 = np.mean(avg_h1[43:43+7])
    h1_10 = np.mean(avg_h1[50:50+7])
    h1_11 = np.mean(avg_h1[57:])

    h2_00 = np.mean(avg_h2[36:36+7])
    h2_01 = np.mean(avg_h2[43:43+7])
    h2_10 = np.mean(avg_h2[50:50+7])
    h2_11 = np.mean(avg_h2[57:])

    h12_00 = np.mean(avg_h12[36:36+7])
    h12_01 = np.mean(avg_h12[43:43+7])
    h12_10 = np.mean(avg_h12[50:50+7])
    h12_11 = np.mean(avg_h12[57:])

    # std_arr = np.array( std_h2_00, std_h2_01, std_h2_10, std_h2_11, std_h12_00, std_h12_01, std_h12_10, std_h12_11])
    # plt.plot(std_arr)
    # plt.show()

    # Substract avg of all traces

    mean_h1 = (h1_00+h1_10+h1_01+h1_11)/4
    mean_h2 = (h2_00+h2_01+h2_10+h2_11)/4
    mean_h12 = (h12_00+h12_11+h12_01+h12_10)/4

    avg_h1 -= mean_h1
    avg_h2 -= mean_h2
    avg_h12 -= mean_h12

    scale_h1 = (h1_00+h1_10-h1_01-h1_11)/4
    scale_h2 = (h2_00+h2_01-h2_10-h2_11)/4
    scale_h12 = (h12_00+h12_11-h12_01-h12_10)/4

    avg_h1 = (avg_h1)/scale_h1
    avg_h2 = (avg_h2)/scale_h2
    avg_h12 = (avg_h12)/scale_h12
    # dived by scalefactor

    # key for next step
    h1_00 = np.mean(avg_h1[36:36+7])
    h1_01 = np.mean(avg_h1[43:43+7])
    h1_10 = np.mean(avg_h1[50:50+7])
    h1_11 = np.mean(avg_h1[57:])

    h2_00 = np.mean(avg_h2[36:36+7])
    h2_01 = np.mean(avg_h2[43:43+7])
    h2_10 = np.mean(avg_h2[50:50+7])
    h2_11 = np.mean(avg_h2[57:])

    h12_00 = np.mean(avg_h12[36:36+7])
    h12_01 = np.mean(avg_h12[43:43+7])
    h12_10 = np.mean(avg_h12[50:50+7])
    h12_11 = np.mean(avg_h12[57:])

    std_h1_00 = np.std(avg_h1[36:36+7])
    std_h1_01 = np.std(avg_h1[43:43+7])
    std_h1_10 = np.std(avg_h1[50:50+7])
    std_h1_11 = np.std(avg_h1[57:])

    std_h2_00 = np.std(avg_h2[36:36+7])
    std_h2_01 = np.std(avg_h2[43:43+7])
    std_h2_10 = np.std(avg_h2[50:50+7])
    std_h2_11 = np.std(avg_h2[57:])

    std_h12_00 = np.std(avg_h12[36:36+7])
    std_h12_01 = np.std(avg_h12[43:43+7])
    std_h12_10 = np.std(avg_h12[50:50+7])
    std_h12_11 = np.std(avg_h12[57:])

    std_h1 = np.mean([std_h1_00, std_h1_01, std_h1_10, std_h1_11])
    std_h2 = np.mean([std_h2_00, std_h2_01, std_h2_10, std_h2_11])
    std_h12 = np.mean([std_h12_00, std_h12_01, std_h12_10, std_h12_11])
    std_arr = np.array([std_h1_00, std_h1_01, std_h1_10, std_h1_11, std_h2_00, std_h2_01,
                        std_h2_10, std_h2_11, std_h12_00, std_h12_01, std_h12_10, std_h12_11])
    fac = np.mean([std_h1, std_h2, std_h12])
    avg_h1 *= fac/std_h1
    avg_h2 *= fac/std_h2
    avg_h12 *= fac/std_h12

    h1_00 = np.mean(avg_h1[36:36+7])
    h1_01 = np.mean(avg_h1[43:43+7])
    h1_10 = np.mean(avg_h1[50:50+7])
    h1_11 = np.mean(avg_h1[57:])

    h2_00 = np.mean(avg_h2[36:36+7])
    h2_01 = np.mean(avg_h2[43:43+7])
    h2_10 = np.mean(avg_h2[50:50+7])
    h2_11 = np.mean(avg_h2[57:])

    h12_00 = np.mean(avg_h12[36:36+7])
    h12_01 = np.mean(avg_h12[43:43+7])
    h12_10 = np.mean(avg_h12[50:50+7])
    h12_11 = np.mean(avg_h12[57:])

    self.measurements_tomo = (
        np.array([avg_h1[0:36], avg_h2[0:36],
                  avg_h12[0:36]])).flatten()  # 108 x 1
    # get the calibration points by averaging over the five measurements
    # taken knowing the initial state we put in
    self.measurements_cal = np.array(
        [h1_00, h1_01, h1_10, h1_11,
         h2_00, h2_01, h2_10, h2_11,
         h12_00, h12_01, h12_10, h12_11])


	def _calibrate_betas(self):
        """
        calculates betas from calibration points for the initial measurement
        operator

        Betas are ordered by B0 -> II B1 -> IZ etc(binary counting)
        <0|Z|0> = 1, <1|Z|1> = -1

        Keyword arguments:
        measurements_cal --- array(2 ** n_qubits) should be ordered
            correctly (00, 01, 10, 11) for 2 qubits
        """
        cal_matrix = np.zeros((self.n_states, self.n_states))
        # get the coefficient matrix for the betas
        for i in range(self.n_states):
            for j in range(self.n_states):
                # perform bitwise AND and count the resulting 1s
                cal_matrix[i, j] = (-1)**(bin((i & j)).count("1"))
        # invert solve the simple system of equations
        betas = np.zeros(12)
        betas[0:4] = np.dot(np.linalg.inv(cal_matrix), measurements_cal[0:4])
        betas[4:8] = np.dot(np.linalg.inv(cal_matrix), measurements_cal[4:8])
        betas[8:] = np.dot(np.linalg.inv(cal_matrix), measurements_cal[8:12])
        return betas

    def expectation_value_calculation_IdenZ(self):

    	betas = _calibrate_betas(self.measurements_cal)
    	#inverting the unprimed beta matrix
    	#up is unprimed
    	beta_0_up =betas[0] 
    	beta_1_up =betas[1]
    	beta_2_up =betas[2]
    	beta_3_up =betas[3]

    	beta_matrix_up = np.array([[beta_0_up,beta_1_up,beta_2_up,beta_3_up],
    									[beta_0_up,-1*beta_1_up,beta_2_up,-1*beta_3_up],
    									[beta_0_up,beta_1_up,-1*beta_2_up,-1*beta_3_up],
    									[beta_0_up,-1*beta_1_up,-1*beta_2_up,beta_3_up]])
    	#assuming 0:4 are 
    	expect_value_IdenZ_up = np.dot(np.linalg.inv(beta_matrix_up), measurements_tomo[0:4])

    	#inverting the primed beta matrix
    	#p is primed
    	beta_0_p =betas[4] 
    	beta_1_p =betas[5]
    	beta_2_p =betas[6]
    	beta_3_p =betas[7]

    	beta_matrix_p = np.array([[beta_0_p,beta_1_p,beta_2_p,beta_3_p],
    									[beta_0_p,-1*beta_1_p,beta_2_p,-1*beta_3_p],
    									[beta_0_p,beta_1_p,-1*beta_2_p,-1*beta_3_p],
    									[beta_0_p,-1*beta_1_p,-1*beta_2_p,beta_3_p]])
    	#assuming 0:4 are 
    	expect_value_IdenZ_p = np.dot(np.linalg.inv(beta_matrix_p), measurements_tomo[0:4])

    	#inverting the unprimed beta matrix
    	#up is unprimed
    	beta_0_pp =betas[8] 
    	beta_1_pp =betas[9]
    	beta_2_pp =betas[10]
    	beta_3_pp =betas[11]

    	beta_matrix_pp = np.array([[beta_0_pp,beta_1_pp,beta_2_pp,beta_3_pp],
    									[beta_0_pp,-1*beta_1_pp,beta_2_pp,-1*beta_3_pp],
    									[beta_0_pp,beta_1_pp,-1*beta_2_pp,-1*beta_3_pp],
    									[beta_0_pp,-1*beta_1_pp,-1*beta_2_pp,beta_3_pp]])
    	#assuming 0:4 are 
    	expect_value_IdenZ_pp = np.dot(np.linalg.inv(beta_matrix_pp), measurements_tomo[0:4])

    	#take the mean of calculated expectation values of II, IZ, ZI, ZZ
    	#for three different beta vectors
    	expect_value_IdenZ = np.mean( np.array([expect_value_IdenZ_up.flatten(), 
    											expect_value_IdenZ_p.flatten(), 
    											expect_value_IdenZ_pp.flatten()]), 
    											axis=0 )


    	return expect_value_IdenZ

    def expectation_value_calculation_XX(self):

    	betas = _calibrate_betas(self.measurements_cal)
    	expect_value_XX_up = (measurements_tomo[4] + measurements_tomo[5])/betas[3]
    	expect_value_XX_p = (measurements_tomo[4] + measurements_tomo[5])/betas[7]
    	expect_value_XX_pp = (measurements_tomo[4] + measurements_tomo[5])/betas[11]
    	expectation_value_XX = (expect_value_XX_up + expect_value_XX_p + expect_value_XX_pp /3)

    	return expectation_value_XX

	def expectation_value_calculation_XX(self):

		betas = _calibrate_betas(self.measurements_cal)
		expect_value_YY_up = (measurements_tomo[6] + measurements_tomo[7])/betas[3]
    	expect_value_YY_p = (measurements_tomo[6] + measurements_tomo[7])/betas[7]
    	expect_value_YY_pp = (measurements_tomo[6] + measurements_tomo[7])/betas[11]
    	expectation_value_YY = (expect_value_XX_up + expect_value_XX_p + expect_value_XX_pp /3)
		
		return expectation_value_YY


    def execute_expectation_value_calculation(self):

    	expect_values = np.zeros(6)
    	expect_values[0:4]  = expectation_value_calculation_IdenZ
    	expect_values[4]	= expectation_value_calculation_XX
    	expect_values[5]	= expectation_value_calculation_YY
    	return expect_values






