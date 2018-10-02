'''
Handles post processing of the measurement data
Leaky integrate and fire always implemented, turn on and off by setting alpha.
Train data with ridge regression/pseudo-inverse
Calulate fitness/error

!for now all functions support only one output node
'''
import numpy as np
from modules.classifiers import perceptron as classifier

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
    '''This implements the fitness function
    F = par[0] * m / (sqrt(r) + par[3] * abs(c)) + par[1] / r + par[2] * Q
    where m,c,r follow from fitting x = m*target + c to minimize r
    and Q is the fitness quality as defined by Celestine in his thesis
    appendix 9'''


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
        return F
    #F += (min(x1) - max(x0)) * 20
    # F = F - clipcounter *  0.1
    #clipcounter = 0            
    #return F

def fitnessREvolution(x, target, W, par):
    '''This implements the regularized fitness function
    F = par[0] * m / (sqrt(r) + par[3] * abs(c)) + par[1] / r + par[2] * Q
    where m,c,r follow from fitting x = m*target + c to minimize r
    and Q is the fitness quality as defined by Celestine in his thesis
    appendix 9'''


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
        Q = np.abs(min(x1) - max(x0)) #/ (max(x1) - min(x0) + abs(min(x0)))

    F = par[0] * m / (res**(.5) + par[3] * abs(c)) + par[1] / res + par[2] * Q
    ## REgularize 
#    print('Offset c = ',c)
    F = F - 0.1*m**2
    clipcounter = 0
    for i in range(len(x_weighed)):
        if(abs(x_weighed[i]) > 3.1*10):
            clipcounter = clipcounter + 1
            F = -100
    # F = F - clipcounter *  0.1
    clipcounter = 0            
    return F

    
def fitnessEvolutionSpiral(x, target, W, par):
    '''This implements the fitness function
    F = par[0] * m / (sqrt(r) + par[3] * abs(c)) + par[1] / r + par[2] * Q
    where m,c,r follow from fitting x = m*target + c to minimize r
    and Q is the fitness quality as defined by Celestine in his thesis
    appendix 9'''

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
    signsum = abs(sum(np.sign(Difference)))/(len(x)-1)

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


def fitnessEvolutionClassif(x, par):
    y = np.zeros(len(x))
    z = np.zeros(len(y)-1)
    var = np.zeros(len(y))
    for i in range(len(x)):
        y[i] = np.average(x[i])
        var[i] = np.std(x[i])
    y = sorted(y)

    for j in range(len(y)-1):
        z[j] = abs(y[j+1]-y[j])

    Difference = np.max(y)-np.min(y)  
    NoiseContr = np.exp(-np.average(var)/(0.01*Difference))

    F = par[0] * np.amin(z) + par[1] * np.average(z) + NoiseContr * par[2]
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*10):
                return -100
    return F

def fitnessEvolutionClassifSep(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*10):
                return -100

    F = 0
    y = np.average(x,axis = 1)
    ind = np.argsort(y)
    for i in np.arange(1,len(ind)-1):
        a = ind[i]
        b = ind[i+1]
        c = ind[i-1]
        if np.min(x[a])>np.max(x[c]) and np.max(x[a])<np.min(x[b]):
            F = F+1
    return F

def fitnessEvolutionSingelinputrecongnition(x, optimal_input, par):
    y = np.zeros(len(x))
    z = np.zeros(len(y))

    for i in range(len(x)):
        y[i] = np.average(x[i])
    
    for n in range(len(y)):
    	z[n] = abs(y[optimal_input]-y[n])
    z = np.sort(z)[1:]

    F = par[0] * np.amin(z) + par[1] * np.average(z)
    for i in range(len(x)): 
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*10):
                return -100
    return F

def fitnessmaxmin(x, optimal_input, par):
    y = np.zeros(len(x))
    z = np.zeros(len(y))

    for i in range(len(x)):

        y[i] = np.average(x[i])

    indeces = np.argsort(y)
    c = y[optimal_input]

    if y[optimal_input] >= 0:
	    y[optimal_input] = -100
	    F = (c-np.amax(y))/abs(c) + abs(c)/3000
    else:
	    y[optimal_input] = 100
	    F = (np.amin(y)-c)/abs(c) + abs(c)/3000

    for i in range(len(x)): 
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*100):
                return -100

    return F

def fitnessvariance(x, optimal_input, par):
    y = np.zeros(len(x))
    indistd = np.zeros(len(x))
    z = np.zeros(len(y))
    var = np.zeros(len(y))
    for i in range(len(x)):
        y[i] = np.average(x[i])
        var[i] = np.std(x[i])
    others = np.delete(y, optimal_input)
    stdothers = np.delete(var, optimal_input) 
    

    averagevar = np.average(var)

    average = np.average(others)
    variance = np.std(others)

    Difference = np.max(y)-np.min(y)  
    NoiseContr = np.exp(-np.average(var)/(0.01*Difference))
    aveTemp=0
    nN=0.00001

    if y[optimal_input] >= 0:
        for i in range(len(others)):
            if others[i] >= 0:
               aveTemp=aveTemp+others[i]
               nN=nN+1
        aveTemp=aveTemp/nN
        if y[optimal_input] >= np.amax(others):
            F = (par[0]*abs(y[optimal_input])**2)/(par[1]*abs(aveTemp)+par[2]*np.std(others))+par[3]*(abs(y[optimal_input])-abs(np.amax(others)))
        else:
            F = (par[0]*abs(y[optimal_input])**2)/(par[1]*abs(aveTemp)+par[2]*np.std(others))
    else:
        for i in range(len(others)):
            if others[i] <= 0:
               aveTemp=aveTemp+others[i]
               nN=nN+1
        aveTemp=aveTemp/nN
        if y[optimal_input] <= np.amin(others):
            F = (par[0]*abs(y[optimal_input])**2)/(par[1]*abs(aveTemp)+par[2]*np.std(others))+par[3]*(abs(y[optimal_input])-abs(np.amin(others)))
        else:
            F = (par[0]*abs(y[optimal_input])**2)/(par[1]*abs(aveTemp)+par[2]*np.std(others))
    + NoiseContr * par[2]
    for i in range(len(x)): 
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*10):
                return -100
    return F

def fitnesssigndiff(x, optimal_input, par):
    y = np.zeros(len(x))
    indistd = np.zeros(len(x))
    z = np.zeros(len(y))
    var = np.zeros(len(y))
    for i in range(len(x)):
        y[i] = np.average(x[i])
        var[i] = np.std(x[i])
    others = np.delete(y, optimal_input)
    stdothers = np.delete(var, optimal_input) 
    
    averagevar = np.average(var)

    average = np.average(others)
    variance = np.std(others)

    Difference = np.max(y)-np.min(y)  
    NoiseContr = np.exp(-np.average(var)/(0.01*Difference))
    aveTemp=0
    nN=0.00001
    
    # counterpos = 0
    # for i in range(len(others)):
    #         if others[i] >= 0:
    #             counterpos = counterpos+1

    # counterneg = 0
    # for i in range(len(others)):
    #         if others[i] <= 0:
    #             counterneg = counterneg+1

    if y[optimal_input] >= 0:
        # aveTemp = np.zeros(counterneg)
        # for i in range(len(others)):
        #     if others[i] >= 0:
        #        aveTemp[i] = others[i]
        #        nN=nN+1
        # aveTemp=aveTemp/nN

        F = (y[optimal_input]-np.amax(others))

    else:
        # aveTemp = np.zeros(counterneg)
        # for i in range(len(others)):
        #     if others[i] <= 0:
        #        aveTemp[i] = others[i]
        #        nN=nN+1
        # aveTemp=aveTemp/nN

        F = (np.amin(others)-y[optimal_input])

    for i in range(len(x)): 
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*10):
                return -100
    return F


def fitnesssquareinput(x, optimal_input, par):
    y = np.zeros(len(x))
    z = np.zeros(len(y))
    var = np.zeros(len(y))
    for i in range(len(x)):
        y[i] = np.average(x[i])
        var[i] = np.std(x[i])
    others = np.delete(y, optimal_input)
    stdothers = np.delete(indistd, optimal_input)
    
    average = np.average(others)
    variance = np.std(others)

    Difference = np.max(y)-np.min(y)  
    NoiseContr = np.exp(-np.average(var)/(0.01*Difference))

    F = par[0]*(abs(y[optimal_input])**2-indistd(optimal_input))/(abs(average)*variance)
    # + NoiseContr * par[2]

    for i in range(len(x)): 
        for j in range(len(x[0])):
            if(abs(x[i][j])>3.1*100):
                return -100
   
    return F

def alphaFit(x, target, W, par):
    
    F = fitnessEvolution(x, target, W, par)
    #extract fit data with weights W
    indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)
    x_weighed = np.empty(len(indices))
    target_weighed = np.empty(len(indices))
    for i in range(len(indices)):
    	x_weighed[i] = x[indices[i]]
    	target_weighed[i] = target[indices[i]]
    x_weighed = x_weighed[:,np.newaxis]
    target_weighed = target_weighed[:,np.newaxis]
    
    alpha = classifier(x_weighed,target_weighed)
    
    return alpha*F

def fitnessNegMSE(x, target, W, par):
    #extract fit data with weights W
    indices = np.argwhere(W)  #indices where W is nonzero (i.e. 1)

    x_weighed = np.empty(len(indices))
    target_weighed = np.empty(len(indices))
    for i in range(len(indices)):
    	x_weighed[i] = x[indices[i]]
    	target_weighed[i] = target[indices[i]]
        
    negMSE = -np.mean((target_weighed-x_weighed)**2)
    
    return negMSE

    
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

