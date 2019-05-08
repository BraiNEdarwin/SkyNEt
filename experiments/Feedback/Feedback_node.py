# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:21:06 2019

@author: Jardi
"""

import numpy as np
import matplotlib.pyplot as plt
from SkyNEt.modules.Nets.resNode import resNode

## constants
N = 1000
input_electrode = 6
feedback_electrode = 0
delay = 1
init_value = None

if input_electrode == 0 or input_electrode == 1:
    vlow = -0.8
    vhigh = 0.2
else:
    vlow = -1.1
    vhigh = 0.7

## input signal
inpt = np.repeat(np.random.uniform(vlow, vhigh, int(N/5)), 5)
inpt = np.zeros(N)

## Load neural net
main_dir = r'C:/Users/Jardi/Desktop/BachelorOpdracht/NNModel/'
data_dir = '24-04-21h48m_NN_lossMSE-d20w90-lr0.003-eps500-mb2048-b10.9-b20.75.pt'
node = resNode(main_dir+data_dir)

## Initialise cvs without feedback
node.init_cvs(input_electrode)

## predict output without feedback
prediction1, inputs = node.outputs(inpt)

## add feedback
node.add_feedback(feedback_electrode, delay, init_value)

## predict output with feedback
prediction2, inputs_feedback = node.outputs(inpt)

## get memory capacity
mc0 = node.get_MC(inpt, prediction1)
mc = node.get_MC(inpt, prediction2)
#print(MC0, MC)

## plot stuff
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(inpt)
plt.xlabel('Time step')
plt.ylabel('Input voltage (V)')
plt.title('Input (electrode 6)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(prediction1)
plt.plot(prediction2)
plt.xlabel('Time step')
plt.ylabel('Output current (nA)')
plt.title('Output')
plt.grid(True)
plt.legend(('Without feedback', 'With feedback'), loc='best')

plt.tight_layout()

plt.figure()
plt.plot(np.linspace(1, len(mc), len(mc)), mc)
plt.grid(True)
plt.tight_layout()