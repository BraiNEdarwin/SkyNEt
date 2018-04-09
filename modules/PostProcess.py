'''
Handles post processing of the measurement data
Leaky integrate and fire always implemented, turn on and off by setting alpha.
Train data with ridge regression/pseudo-inverse
Calulate fitness/error

!for now all functions support only one output node
'''
import numpy as np


def leakyIntegrateFire(x, alpha):
    for i in range(1, len(x)):
        x[i] = (1 - alpha) * x[i] + alpha * x[i - 1]


def trainNetwork(method, output, target, a):
    if (method == 'ridgeRegression'):
        wOut = np.dot(np.transpose(output), target) / \
            (np.dot(np.transpose(output), output) + a ** 2)
        return wOut * output

    if (method == 'pseudoInverse'):
        wOut = np.transpose(np.dot(np.linalg.pinv(output), target))
        return wOut * output


def fitness(x, target):
    return 1 / ((np.linalg.norm(x - target, 2)) ** 2 * (1 / len(x)))

def fitnessEvolution(x, target, W):
    
    #extract fit data with weights W
    indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)
    indices = indices[0]  #extract np array

    x_weighed = np.empty(len(indices))
    target_weighed = np.empty(len(indices))
    for i in range(len(indices)):
    	x_weighed = x[i]
    	target_weighed = target[i]


	#fit x = m * target + c to minimize res
    A = np.vstack([target, np.ones(len(indices))]).T  #write x = m*target + c as x = A*target 
    m, c = np.linalg.lstsq(A, x_weighed)[0]	
    res = np.linalg.lstsq(A, x_weighed)[1]
    
    return target_weighed
    #determine fitness quality
