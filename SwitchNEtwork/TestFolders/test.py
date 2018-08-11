#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:03:22 2018

@author: renhori
"""
import numpy as np
import time

#define voltage range and initialize stuff
#right now it's set for 1.5 to -1.5 in a step of 5mV
voltrange = []
a = np.zeros((4,1,301))
currentlist = np.zeros((8,7,2,1204))
bytelist=[0,0,0,0,0,0,0,0]

first =(np.linspace(0,-1.5, 301))
a[0] = first
second= (np.linspace(-1.5, 0,301))
a[1] = second
third= (np.linspace(0, 1.5, 301))
a[2] = third
fourth = (np.linspace(1.5, 0, 301))
a[3] = fourth


for b in range(4):
    for c in range(301):
        voltrange.append(a[b][0][c])

#a corresponds to the device number
for a in range(len(currentlist)):
    #open the input port of only the particular device
    bytelist[0] = 2**(7-a)
    #b corresponds to the connection from the electrode 1 to that electrode
    for b in range(len(currentlist[a])):
        bytelist[b+1] = 2**(7-a)
        #for the range of voltage
        for c in range (len(voltrange)):
            #get current
            #current=[]
            #keithley.volt.set(voltrange[c])
            #time.sleep(0.01)
            print(voltrange[c])
            #current[c] = keithley.curr()
            #currentlist[a][b][0][c] = voltrange[c]
            #currentlist[a][b][1][c] = current[c]
            
        