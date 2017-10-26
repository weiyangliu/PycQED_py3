import sys  
#custom made toolboxes
sys.path.append('D:\\Repository\\PycQED_py3\\pycqed')


from analysis import tomography_toolbox as tomography
import os
import time
from imp import reload
from matplotlib import pyplot as plt
import numpy as np
from analysis import measurement_analysis as MA 
from analysis import ramiro_analysis as RA
from analysis import thresholding_toolbox as thresholding
from analysis import fitting_models as fit_mods
import lmfit
import scipy as scipy
try:
    import qutip as qtp
except ImportError as e:
    logging.warning('Could not import qutip, tomo code will not work')
import itertools

class TomoPrep(BaseDataAnalysis):
	"""
	class to prepare the necessary inputs for tomography_V2.
	Does thresholding and coincidence counting on measured voltages
	to convert them into counts.
	Constructs measurement operators based on callibration counts.
	Counts and callibrated measurement operators 


	""" 
	#TODO: Improve init

	def __init__(self,):



	def histogram_shots(shots):
    	hist, bins = np.histogram(shots, bins=100, normed=True)
    	# 0.7 bin widht is a sensible default for plotting
    	centers = (bins[:-1] + bins[1:]) / 2
    	return hist, bins, centers
	
	def fit_data(hist, centers):
	    model = fit_mods.DoubleGaussModel
	    params = model.guess(model, hist, centers)
	    fit_res = model.fit(data=hist, x=centers, params=params)
	    return fit_res

	def make_figures(hist, centers, show_guess=False):
	    fit_res = fit_data(hist,centers)
	    fig, ax = plt.subplots(figsize=(5, 3))
	    width = .7 * (centers[1]-centers[0])
	  

	    x_fine = np.linspace(min(centers),
	                         max(centers), 1000)
	    # Plotting the data
	    plt.figure()
	    plt.subplot(211)
	    ax.bar(centers, hist, align='center', width=width, label='data')
	    ax.plot(x_fine, fit_res.eval(x=x_fine), label='fit', c='r')
	    if show_guess:
	        ax.plot(x_fine, fit_res.eval(
	        	x=x_fine, **fit_res.init_values), label='guess', c='g')
	        ax.legend(loc='best')

	    # Prettifying the plot
	    ax.ticklabel_format(useOffset=False)
	    ax.set_ylabel('normalized counts')
	    plt.show()
	    return fit_re
	def make_s_curve(N, sigma, mu,start,stop):
    	#numerically normalize the gaussian so that the area under the curve is 1
    
    	#use analytical formula for s-curve
    	x = np.linspace(start,stop,100)
    	z = np.divide(x - mu,np.sqrt(2)**sigma) 
    	erf1 = N**sigma**scipy.special.erf(z)
    	return erf
    def get_thresholds(shots_q0, shots_q1, make_figures_flag = False):

		#get all shots for qubits individually
		shots_q0 = shots_q0.flatten()
		shots_q1 = shots_q1.flatten()


		#For q0 Fit Two gaussians 
		hist1, bins1, centers1 = histogram_shots(shots_q0)
		fit_res1 = fit_data(hist1, centers1)
		if make_figures_flag = True:
			fr1 = make_figures(hist1,centers1)
		#make two s-curves
		erf1_q0 = make_s_curve(fr1.best_values['A_amplitude'],
		                       fr1.best_values['A_sigma'],
		                       fr1.best_values['A_center'],
		                       np.min(shots_q0),
		                       np.amax(shots_q0))
		erf2_q0 = make_s_curve(fr1.best_values['B_amplitude'],
		                       fr1.best_values['B_sigma'],
		                       fr1.best_values['B_center'],
		                       np.min(shots_q0),
		                       np.amax(shots_q0))
		#Plot the s-curve
		x1 = np.linspace(np.min(shots_q0),np.amax(shots_q0),100)
		#extract_threshold
		th0 = x1[np.argmax(np.subtract(erf1_q0,erf2_q0))]
		# plt.figure(1)
		# plt.subplot(211)
		# plt.plot(x1,erf1_q0)
		# plt.xlim([np.min(shots_q0),np.amax(shots_q0)])
		# plt.plot(x1,erf2_q0)
		# plt.xlim([np.min(shots_q0),np.amax(shots_q0)])
		

		# plt.axvline(th0, color='r')
		# plt.show()               
		                
		#For q1 Fit Two gaussians 
		hist2, bins2, centers2 = histogram_shots(shots_q1)
		fit_res2 = fit_data(hist2, centers2)
		fr2 = make_figures(hist2,centers2)
		#make two s-curves
		erf1_q1 = make_s_curve(fr2.best_values['A_amplitude'],
		                       fr2.best_values['A_sigma'],
		                       fr2.best_values['A_center'],
		                       np.min(shots_q1),
		                       np.amax(shots_q1))
		erf2_q1 = make_s_curve(fr2.best_values['B_amplitude'],
		                       fr2.best_values['B_sigma'],
		                       fr2.best_values['B_center'],
		                       np.min(shots_q1),
		                       np.amax(shots_q1))
		#Plot the s-curve
		x2 = np.linspace(np.min(shots_q1),np.amax(shots_q1),100)
		th1 = x2[np.argmax(np.subtract(erf1_q1,erf2_q1))]
		# plt.subplot(212)
		# plt.plot(x2,erf1_q1)
		# plt.xlim([np.min(shots_q1),np.amax(shots_q1)])
		# plt.plot(x2,erf2_q1)
		# plt.xlim([np.min(shots_q1),np.amax(shots_q1)])
		# #extract_threshold
		

		# plt.axvline(th1, color='r')
		# plt.show()               

    	return th0, th1

    def coincidence_counting_tomo(th0, th1, shots_q0, shots_q1):
    
	    """
	    Takes as an input the thresholds and All shots for both qubits
	    Counts the incidents of qubits ending in 00, 01, 10, 11
	    Returns unnormalized counts
	    """
	   # thresholding or 2D - binning
	    #determine which one is zero which one is one
	    side_0_q0 = np.where(np.mean(shots_q0[36:36+7,:])<th0,1,-1)
	    side_0_q1 = np.where(np.mean(shots_q1[36:36+7,:])<th1,1,-1)
	    
	    #use only tomo data
	    shots_q0 = shots_q0[:36,:]
	    shots_q1 = shots_q1[:36,:]
	    
	    #threshold all data
	    counts_q0 = np.where(shots_q0<th0*side_0_q0,0,1).astype('int')
	    counts_q1 = np.where(shots_q1<th1*side_0_q1,0,1).astype('int')
	    # Coincidence counting.
	    counts_00 = np.sum(np.where(np.logical_and(counts_q0==0,counts_q1==0),1,0), axis=1)
	    counts_10 = np.sum(np.where(np.logical_and(counts_q0==1,counts_q1==0),1,0), axis=1)
	    counts_01 = np.sum(np.where(np.logical_and(counts_q0==0,counts_q1==1),1,0), axis=1)
	    counts_11 = np.sum(np.where(np.logical_and(counts_q0==1,counts_q1==1),1,0), axis=1)
    	return np.transpose(np.array([counts_00, counts_01, counts_10, counts_11]))
	def coincidence_counting_calibration(th0, th1, shots_q0, shots_q1):
    
		"""
		Takes as an input the thresholds and all shots(convert it to all ) for both qubits
		Counts the incidents of qubits ending in 00, 01, 10, 11
		Returns unnormalized counts
		"""

		# Keep only calibration shots
		shots_q0 = shots_q0[36:,:]
		shots_q1 = shots_q1[36:,:]


		# thresholding or 1D - binning

		side_0_q0 = np.where(np.mean(shots_q0[:7])<th0,1,-1)
		side_0_q1 = np.where(np.mean(shots_q1[:7])<th1,1,-1)
		#Thresholding
		counts_q0 = np.where(shots_q0<th0*side_0_q0,0,1).astype('int')
		counts_q1 = np.where(shots_q1<th1*side_0_q1,0,1).astype('int')
		# Counting Coincidences
		counts_00 = np.sum(np.where(np.logical_and(counts_q0==0,counts_q1==0),1,0), axis=1)
		counts_01 = np.sum(np.where(np.logical_and(counts_q0==1,counts_q1==0),1,0), axis=1)
		counts_10 = np.sum(np.where(np.logical_and(counts_q0==0,counts_q1==1),1,0), axis=1)
		counts_11 = np.sum(np.where(np.logical_and(counts_q0==1,counts_q1==1),1,0), axis=1)

		count_stack = (np.transpose(np.array([counts_00, counts_01, counts_10, counts_11])))
		count_stack_reshaped = np.sum(np.vsplit(count_stack,4),axis =1)

    	return count_stack_reshaped  


    def construct_measurement_operators(count_array):
		c_total = np.sum(count_array)
		normalized_counts = (count_array/(c_total/4))
		M_bins_coeff = normalized_counts
		M_bins = [qt.Qobj(np.diag(M_bins_coeff[i,:]), dims=[[2,2],[2,2]]) for i in range(0,4)]
    
    	return M_bins


    def assemble_input_for_tomography(shots_q0, shots_q1):
		#call functions that eventually return the threshold
		th0,th1 = get_thresholds(shots_q0,shots_q1)


		#call coincidence counting for calibration measurement
		calibration_counts= coincidence_counting_calibration(th0, th1, shots_q0, shots_q1)
		print(calibration_counts)
		comp_projectors = construct_measurement_operators(calibration_counts)


		counts_tomo= coincidence_counting_tomo(th0,th1, shots_q0, shots_q1)
        
    	return comp_projectors,counts_tomo
    

