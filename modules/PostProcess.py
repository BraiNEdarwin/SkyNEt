'''
Handles post processing of the measurement data
Leaky integrate and fire always implemented, turn on and off by setting alpha.
Train data with ridge regression/pseudo-inverse
Calulate fitness/error

!for now all functions support only one output node
'''
import numpy as np


def leakyIntegrateFire(x, alpha):
	for i in range(1,len(x)):
		x[i] = (1 - alpha) * x[i] + alpha * x[i-1]


def trainNetwork(method, output, target, a):
	if (method == 'ridgeRegression'):
		wOut = np.dot(np.transpose(output), target) / ( np.dot(np.transpose(output), output) +  a^2)
		return wOut * output

	if (method == 'pseudoInverse'):
		wOut = np.transpose( np.dot(np.linalg.pinv(output), target ) )
		return wOut * output


def fitness(x, target):
	return np.linalg.norm(x - target, 2)


