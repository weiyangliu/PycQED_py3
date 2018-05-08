from .base_lutman import Base_LutMan
from pycqed.measurement.waveform_control_CC import waveform as wf
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import copy as copy
from pycqed.measurement import detector_functions as det

class Base_Acq_LutMan(Base_LutMan):
	    def __init__(self, name, num_res=2, feedline_number: int=0,**kw):
        self.add_parameter('weight_nr_samples', unit='samples', vals=vals.Ints(0,2**20), #taking arbitrarily large max val
                           parameter_class=ManualParameter,
                           initial_value=4096)
        self.add_parameter('averages', unit='samples', vals=vals.Ints(0,2**20), #taking arbitrarily large max val
                           parameter_class=ManualParameter,
                           initial_value=2*11)
        self.add_parameter('digitized', vals=vals.Bool(),
                           initial_value=False,
                           parameter_class=ManualParameter)
        self.add_parameter('integration_length'.format(res), unit='s',
                           vals=vals.Numbers(1e-9, 8000e-9),
                           parameter_class=ManualParameter,
                           initial_value=2000e-9)
	    self.add_parameter('weight_type',
	               initial_value='DSB',
	               vals=vals.Enum('SSB', 'DSB', 'optimal', 'optimal IQ'),
	               docstring=ro_acq_docstr,
	               parameter_class=ManualParameter)
        if num_res > 9:
            raise ValueError('At most 9 resonators can be read out.')
        self._num_res = num_res
        self._feedline_number = feedline_number
        if self._feedline_number==0:
            self._resonator_codeword_bit_mapping=[0,2,3,5,6]
        elif self._feedline_number==1:
            self._resonator_codeword_bit_mapping=[1,4]
        else:
            raise NotImplementedError(
              'hardcoded for feedline 0 and 1 of Surface-7')
        #capping the resonator bit mapping in case a limited number of resonators is used
        self._resonator_codeword_bit_mapping = self._resonator_codeword_bit_mapping[:self._num_res]
        super().__init__(name, **kw)

    def _add_waveform_parameters(self):
        # mixer corrections are done globally, can be specified per resonator
        self.add_parameter('mixer_apply_correction_matrix',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)
        self.add_parameter('mixer_alpha', vals=vals.Numbers(),
                           parameter_class=ManualParameter,
                           initial_value=1.0)
        self.add_parameter('mixer_phi', vals=vals.Numbers(), unit='deg',
                           parameter_class=ManualParameter,
                           initial_value=0.0)
        self.add_parameter('mixer_offs_I', unit='V',
                           parameter_class=ManualParameter, initial_value=0)
        self.add_parameter('mixer_offs_Q', unit='V',
                           parameter_class=ManualParameter, initial_value=0)       

        for res in self._resonator_codeword_bit_mapping:
            self.add_parameter('R{}_modulation'.format(res),
                           vals=vals.Numbers(), unit='Hz',
                           parameter_class=ManualParameter,
                           initial_value=20.0e6)
            self.add_parameter('R{}_opt_weights_I',
                           vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        	self.add_parameter('R{}_opt_weights_Q',
                           vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
                           parameter_class=ManualParameter)
        	self.add_parameter('R{}_digitized_threshold', unit='V',
                           initial_value=0,
                           parameter_class=ManualParameter)
    
    def generate_standard_waveforms(self):
        # Only generate the combinations required/specified in the LutMap
        # RO integration functions
        self._wave_dict = {}
        for res in self._resonator_codeword_bit_mapping:
        	IF = self.get('R{}_modulation'.format(res))
        	# 1. Generate weigt envelopes
            ## DSB weights
            trace_length = self.weight_nr_samples()
	        tbase = np.arange(0, trace_length/self.sampling_rate(), 1/self.sampling_rate())
	        cos = np.array(np.cos(2*np.pi*IF*tbase))
	        sin = np.array(np.sin(2*np.pi*IF*tbase))
	        self._wave_dict['R{}_cos'.format(res)] = cos
	        self._wave_dict['R{}_sin'.format(res)] = sin

	def get_input_average_detector(self, CC, qubit_nr,  **kw):
    	self.prepare_single_qubit_detectors(CC=CC, qubit_nr=qubit_nr)
        return self._input_average_detector

    def get_int_avg_det(self, CC, qubit_nr, **kw):
    	self.prepare_single_qubit_detectors(CC=CC, qubit_nr=qubit_nr) 
    	return self._int_avg_det

    def get_int_avg_det_single(self, CC, qubit_nr, **kw):
    	self.prepare_single_qubit_detectors(CC=CC, qubit_nr=qubit_nr)  
        return self._int_avg_det_single

    def get_int_log_det(self, CC, qubit_nr, **kw):
    	self.prepare_single_qubit_detectors(CC=CC, qubit_nr=qubit_nr)  
        return self._int_log_det


class UHFQC_Acq_LutMan(Base_Acq_LutMan):
	def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.sampling_rate(1.8e9)
        self.weight_nr_samples(4096)

    def load_waveform_onto_instr_lookuptable(self, qubit_nr, regenerate_waveforms: bool=False):
    	if regenerate_waveforms:
    		generate_standard_waveforms()
    	res_nr = self._resonator_codeword_bit_mapping.index(qubit_nr)
    	cos = self._wave_dict['R{}_cos'.format(qubit_nr)]
    	sin = self._wave_dict['R{}_sin'.format(qubit_nr)]
    	weight_I_channel = res_nr
    	if res_nr==8
    		weight_Q_channel=0
    	else:
	    	weight_Q_channel = res_nr + 1 #borrowing integration channel from the next resonator
    	instr=self.instr.get_instr()
    	if self.weight_type() == 'DSB':
    		instr.set('quex_wint_weights_{}_real'.format(weight_I_channel),
	                 np.array(cos))
	    	instr.set('quex_wint_weights_{}_real'.format(weight_Q_channel), 
	                 np.array(sin))
	    	instr.set('quex_rot_{}_real'.format(weight_I_channel), 2.0)
        	instr.set('quex_rot_{}_imag'.format(weight_I_channel), 0.0)
        	instr.set('quex_rot_{}_real'.format(weight_Q_channel), 2.0)
        	instr.set('quex_rot_{}_imag'.format(weight_Q_channel), 0.0)
        elif self.weight_type() == 'SSB':
        	instr.set('quex_wint_weights_{}_real'.format(weight_I_channel),
                 np.array(cos))
        	instr.set('quex_wint_weights_{}_imag'.format(weight_I_channel),
                 np.array(sin))
        	instr.set('quex_wint_weights_{}_real'.format(weight_Q_channel),
                 np.array(sin))
        	instr.set('quex_wint_weights_{}_imag'.format(weight_Q_channel),
                 np.array(cos))
			instr.set('quex_rot_{}_real'.format(weight_I_channel), 1.0)
			instr.set('quex_rot_{}_imag'.format(weight_I_channel), 1.0)
			instr.set('quex_rot_{}_real'.format(weight_Q_channel), 1.0)
			instr.set('quex_rot_{}_imag'.format(weight_Q_channel), -1.0)
		elif 'optimal' in self.weight_type():
			instr.set('quex_wint_weights_{}_real'.format(
                        weight_I_channel),
                        self.get('R{}_opt_weights_I'.format(qubit_nr)))
	        instr.set('quex_wint_weights_{}_imag'.format(
	            		weight_I_channel),
	            		self.get('R{}_opt_weights_Q'.format(qubit_nr)))
	        instr.set('quex_rot_{}_real'.format(
	            weight_I_channel), 1.0)
	        instr.set('quex_rot_{}_imag'.format(
	            weight_I_channel), -1.0) 
	        if self.weight_type() == 'optimal IQ':
	        	instr.set('quex_wint_weights_{}_real'.format(
                    weight_Q_channel),
                        self.get('R{}_opt_weights_I'.format(qubit_nr)))
	        	instr.set('quex_wint_weights_{}_imag'.format(
	            	weight_Q_channel),
	            		self.get('R{}_opt_weights_Q'.format(qubit_nr)))
	        	instr.set('quex_rot_{}_real'.format(
	            	weight_Q_channel), 1.0)
	        	instr.set('quex_rot_{}_imag'.format(
	            	weight_Q_channel), 1.0) 

    def load_waveforms_onto_instr_lookuptable(self, regenerate_waveforms: bool=True):
    	if regenerate_waveforms:
    		self.generate_standard_waveforms():
    	if self.weight_type() in ['SSB', 'DSB', 'optimal IQ']:
    		raise Exception("is only possible for weight_type='optimal'")
    	else:
    		for qubit_nr in self._resonator_codeword_bit_mapping:
    			self.load_waveform_onto_instr_lookuptable(qubit_nr)

   	def prepare_single_qubit_detectors(self, CC, qubit_nr):
   		res_nr = self._resonator_codeword_bit_mapping.index(qubit_nr)
   		weight_I_channel = res_nr
    	UHFQC=self.instr.get_instr()
    	if res_nr==8
    		weight_Q_channel=0
    	else:
	    	weight_Q_channel = res_nr + 1 #borrowing integration channel from the next resonator

   		if self.weight_type() == 'optimal':
            ro_channels = [weight_I_channel]
            result_logging_mode = 'lin_trans'
            if self.digitized():
                result_logging_mode = 'digitized'
            # The threshold that is set in the hardware  needs to be
            # corrected for the offset as this is only applied in
            # software.
            threshold = self.ro_acq_threshold()
            offs = UHFQC.get(
                'quex_trans_offset_weightfunction_{}'.format(weight_I_channel))
            hw_threshold = threshold + offs
            UHFQC.set('quex_thres_{}_level'.format(weight_I_channel), hw_threshold)
        else:
            ro_channels = [weight_I_channel, weight_Q_channel]
            result_logging_mode = 'raw'
        self._input_avg_det = det.UHFQC_input_average_detector(
            UHFQC=UHFQC,
            AWG=CC,
            nr_averages=self.averages(),
            nr_samples=self.weight_nr_samples(), 
            **kw)
        self._int_avg_det = det._UHFQC_integrated_average_detector(
        	UHFQC=UHFQC,
        	AWG=CC,
            channels=ro_channels,
            result_logging_mode=result_logging_mode,
            nr_averages=self.averages(),
            integration_length=self.integration_length(), **kw)
         self._int_avg_det_single = det.UHFQC_integrated_average_detector(
            UHFQC=UHFQC, 
            AWG=CC,
            channels=ro_channels,
            result_logging_mode=result_logging_mode,
            nr_averages=self.averages(),
            real_imag=True, single_int_avg=True,
            integration_length=self.integration_length(), **kw)
         self._int_log_det = det.UHFQC_integration_logging_det(
            UHFQC=UHFQC, 
            AWG=CC,
            channels=ro_channels,
            result_logging_mode=result_logging_mode,
            integration_length=self.integration_length(), **kw)














    
