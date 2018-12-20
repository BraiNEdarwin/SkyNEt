import numpy as np

class experiment_config(object):

	def __init__(self):

	self.filepath = r'F:\test'
	self.name = 'test'

	self.v_low = 0
	self.v_high = 0.1
	self.n_points = 1000
	self.x = np.arange(n_points)
	self.Input = [0]*n_points

	self.Sawtooth = self.Sawtooth


	def Sawtooth(self, v_low, v_high, n_points)
		n_points = n_points/4

		Input1 = np.linspace(v_low, v_high, n_points)	    
		
		Input[0:len(Input1)] = Input1
		Input[len(Input1):(len(Input1)*2)] = Input1
		Input[len(Input1)*2:(len(Input1)*3)] = Input1
		Input[len(Input1)*3:(len(Input1)*4)] = Input1

		return Input 

