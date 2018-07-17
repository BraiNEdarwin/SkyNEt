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

def fitnessEvolution(x, target, W, par):
    #this implements the fitness function
    #F = par[0] * m / (sqrt(r) + par[3] * abs(c)) + par[1] / r + par[2] * Q
    #where m,c,r follow from fitting x = m*target + c to minimize r
    #and Q is the fitness quality as defined by Celestine in his thesis
    #appendix 9


    #extract fit data with weights W
    indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)

    x_weighed = np.empty(len(indices))
    target_weighed = np.empty(len(indices))
    for i in range(len(indices)):
    	x_weighed[i] = x[indices[i]]
    	target_weighed[i] = target[indices[i]]


	#fit x = m * target + c to minimize res
    A = np.vstack([target_weighed, np.ones(len(indices))]).T  #write x = m*target + c as x = A*(m, c)
    m, c = np.linalg.lstsq(A, x_weighed)[0]	
    res = np.linalg.lstsq(A, x_weighed)[1]
    res = res[0]

    #determine fitness quality
    indices1 = np.argwhere(target_weighed)  #all indices where target is nonzero
    x0 = np.empty(0)  #list of values where x should be 0
    x1 = np.empty(0)  #list of values where x should be 1
    for i in range(len(target_weighed)):
        if(i in indices1):
            x1 = np.append(x1, x_weighed[i])
        else:
            x0 = np.append(x0, x_weighed[i])
    if(min(x1) < max(x0)):
        Q = 0
    else:
        Q = (min(x1) - max(x0)) / (max(x1) - min(x0) + abs(min(x0)))

    F = par[0] * m / (res**(.5) + par[3] * abs(c)) + par[1] / res + par[2] * Q
    clipcounter = 0
    for i in range(len(x_weighed)):
        if(abs(x_weighed[i]) > 3.1*10):
            clipcounter = clipcounter + 1
            F = -100
    # F = F - clipcounter *  0.1
    clipcounter = 0            
    return F

def fitnessEvolutionSpiral(x, target, W, par):
    #this implements the fitness function
    #F = par[0] * m / (sqrt(r) + par[3] * abs(c)) + par[1] / r + par[2] * Q
    #where m,c,r follow from fitting x = m*target + c to minimize r
    #and Q is the fitness quality as defined by Celestine in his thesis
    #appendix 9

    #extract fit data with weights W
    indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)

    x_weighed = np.empty(len(indices))
    target_weighed = np.empty(len(indices))
    for i in range(len(indices)):
        x_weighed[i] = x[indices[i]]
        target_weighed[i] = target[indices[i]]

    #find and optimize for only positive or negative differences
    Difference = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        Difference[i] = [i] = x[i+1]-x[i]
    signsum = abs(sum(np.sign(difference)))/(len(x)-1)

    #fit x = m * target + c to minimize res
    A = np.vstack([target_weighed, np.ones(len(indices))]).T  #write x = m*target + c as x = A*(m, c)
    m, c = np.linalg.lstsq(A, x_weighed)[0] 
    res = np.linalg.lstsq(A, x_weighed)[1]
    res = res[0]

        
    F = par[0] * m / (res**(.5) + par[3] * abs(c)) - par[1] * (m-1)^2 + par[2]*signsum / (len(x)-1)
    clipcounter = 0
    for i in range(len(x_weighed)):
        if(abs(x_weighed[i]) > 3.1*10):
            clipcounter = clipcounter + 1
            F = -100
    # F = F - clipcounter *  0.1
    clipcounter = 0            
    return F


def fitnessEvolutionCalssif(x, par):
    y = np.zeros(len(x))
    z = np.zeros(len(y)-1)

    for i in range(len(x)):
        y[i] = np.average(x[i])
    y = sorted(y)

    for j in range(len(y)-1):
        z[j] = abs(y[j+1]-y[j])

    Difference = np.max(x)-np.min(x)  

    F = par[0] * np.amin(z) + par[1] * Difference
    for i in range(len(x)):
        if(abs(x[i])>3.1*10):
            f = -100
    return F

    
def fitnessEvolutionSingelinputrecongnition(x, optimal_input, par):
    y = np.zeros(len(x))
    z = np.zeros(len(y))

    for i in range(len(x)):
        y[i] = np.average(x[i])
    
    for n in range(len(y)):
        z[n] = abs(y[optimal_input]-y[n])

    z[optimal_input] = 100

    F = par[0] * np.amin(z)
    for i in range(len(x)):
        if(abs(x[i])>3.1*10):
            f = -100
    return F
