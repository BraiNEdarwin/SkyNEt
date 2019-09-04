import numpy as np 
import matplotlib.pyplot as plt
import math

def pseudoInput(self, vpoints, vc, numberinputs, points_per_input, slopesize):
    noslope=points_per_input-slopesize
    Input = np.zeros((numberinputs,points_per_input*vc))
    print(Input.shape)
    for num in range(numberinputs):
        for i in range(vc):              
            Input[num,i*points_per_input:i*points_per_input+noslope]=np.ones(noslope)*vpoints[i][num]
            Input[num,i*points_per_input+noslope:(i+1)*points_per_input]=np.linspace(vpoints[i][num], vpoints[i+1][num],slopesize)
    return Input

#define two input voltages and the amount of points you want to measure
numberinputs =2
vc =15
vpoints=  [[-0.5,-0.6],
          [0.5,-0.6],
          [0,-0.8],
          [0, 0.8],
          [-0.5,0.6],
          [0.5,0.6],
          [1,0],
          [0,1],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],
          [2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],[2,0],
          [2,0],
                       
          [0,0]]#to end the measurement on 0V
points_per_input=150
n_points =points_per_input*vc
slopesize= 50

Inp = pseudoInput(1, vpoints, vc, numberinputs, points_per_input, slopesize)

yteach = np.zeros((2**vc,vc))
for i in range(yteach.shape[0]):
    for j in range(yteach.shape[1]):
        yteach[i,vc-1-j]=int(bool(i&0b00000001<<j))
print(yteach)
Output=np.ones((6,n_points))
Output[0] = Inp[0]*3
Output[1] = Inp[0]*2
Output[2] = Inp[1]*6
Output[3] = Inp[1]*8
Output[4] = Inp[0]*-1
Output[5] = Inp[1]*-2
Output[0] = Inp[0]*3
Output[1] = Inp[0]*2
Output[2] = Inp[1]*6
Output[3] = Inp[1]*8
Output[4] = Inp[0]*-1
Output[5] = Inp[1]*-2
Output[0] = Inp[0]*3
Output[1] = Inp[0]*2
Output[2] = Inp[1]*6
Output[3] = Inp[1]*8
Output[4] = Inp[0]*-1
Output[5] = Inp[1]*-2
Output[0] = Inp[0]*3
Output[1] = Inp[0]*2
Output[2] = Inp[1]*6
Output[3] = Inp[1]*8
Output[4] = Inp[0]*-1
Output[5] = Inp[1]*-2

avgout=np.zeros((Output.shape[0],vc))
for i in range(avgout.shape[0]):
    for j in range(avgout.shape[1]):
        avgout[i,j]=np.average(Output[i,j*points_per_input:(j+1)*points_per_input-slopesize])
#print('avgout')
#print(avgout)

#calculate the pseudoinverse of avgout
pseudoinverse = np.linalg.pinv(avgout)
#print('pseudoinverse')
#print(pseudoinverse.shape)
#print(pseudoinverse)
##pseudoinversecheck
checker = np.dot(pseudoinverse,avgout)
#print('unity check')
#print(checker.shape)
#print(checker)
##calculate the weight matrix for yteach
weights = np.dot(yteach,pseudoinverse)
#print('weights')
#print(weights.shape)
#print(weights)
#
#
y=np.dot(weights,avgout)
#print('y')
#print(y.shape)
#print(y)
#calculate fitness
fitness = np.zeros(y.shape[0])


for i in range(1,y.shape[0]-1):
    trueval = []
    falseval = []
    for j in range(y.shape[1]):
        if yteach[i][j] !=0:
            trueval.append(y[i,j])
        else:
            falseval.append(y[i,j])
    #check for correct gate
    if (not trueval or not falseval):
        fitness[i]=0
    else:
        fitness[i]= min(trueval)-max(falseval)
    
print('fitness')
print(fitness)
#plt.figure()
#plt.plot(Inp[0,:], '--', label='input 1')
#plt.plot(Inp[1,:], '--', label='input 2')
#plt.legend()
#plt.title('Boolean measurement 1')
#plt.xlabel('time (ms)')
#plt.ylabel('current (10nA)')
#plt.figure()
#plt.plot(Output[0,:], label ='output 1')
#plt.plot(Output[1,:], label ='output 2')
#plt.plot(Output[2,:], label ='output 3')
#plt.plot(Output[3,:], label ='output 4')
#plt.plot(Output[4,:], label ='output 5')
#plt.plot(Output[5,:], label ='output 6')
#plt.legend()
#plt.title('Boolean measurement 1')
#plt.xlabel('time (ms)')
#plt.ylabel('current (10nA)')

#loopamount= int(np.ceil(2**vc/16))
#for loop in range(loopamount):
#    plt.figure()
#    plt.suptitle('failed configurations')
#    for i in range(4):
#        for j in range(4):
#            if fitness[4*i+j]<0:
#                plt.subplot(4,4,4*i+j+1)
#                plt.plot(yteach[16*loop+4*i+j])
#                plt.plot(y[16*loop+4*i+j])
#                plt.title('fitness = %.2f' % fitness[16*loop+4*i+j])
#    plt.tight_layout()
#    plt.figure()
#    plt.suptitle('succesfull configurations')
#    for i in range(4):
#        for j in range(4):
#            if fitness[4*i+j]>=0:
#                plt.subplot(4,4,4*i+j+1)
#                plt.plot(yteach[16*loop+4*i+j])
#                plt.plot(y[16*loop+4*i+j])
#                plt.title('fitness = %.2f' % fitness[16*loop+4*i+j])
#    plt.tight_layout()        
#plt.show()
