from adwin import ADwin, ADwinError
import sys
import os
import time
import ctypes
import array
import matplotlib.pyplot as plt
import numpy as np

DEVICENUMBER = 1
RAISE_EXCEPTIONS = 1
PROCESSORTYPE = '11'
PROCESS = 'ADbasic1_Pr3.TB1'

Input = np.random.randn(100000)
#x = np.linspace(-100*np.pi, 100*np.pi, 100000)
#Input = 10*np.sin(x)
InputBin = Input * (65536 / 20) + (65536 / 2)
InputBin = InputBin.astype(int)
InputSize = len(InputBin)
ArrayFloat = []

#trail zeros if data is shorter than buffer
if (InputSize < 40003):
    InputBin = np.append(InputBin,np.zeros(40003 - InputSize, int))

try:
    adw = ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
    if os.name == 'posix':
        adw.Boot(str('adwin' + PROCESSORTYPE + '.btl'))
    else:
        adw.Boot('C:\\ADwin\\ADwin' + PROCESSORTYPE + '.btl')

        proc = os.path.abspath(os.path.dirname(sys.argv[0])) + os.sep + PROCESS
        
        adw.Load_Process(proc)
        adw.Set_Processdelay(1, 30000)
        
        #fill the write FIFO
        adw.Fifo_Clear(2)
        empty = adw.Fifo_Empty(2)
        adw.SetFifo_Long(2, list(InputBin[:empty]), empty)
        
        adw.Start_Process(1)
#        time.sleep(0.3)  # Give init time
        
        if (InputSize < 40003):
            j = 0  #counts amount of read values
            while(j < InputSize):
                full = adw.Fifo_Full(1)
                
                #check if read FIFO is empty enough
                if (adw.Fifo_Empty(1) < 5000):
                    print("Houston we've got a read problem." + str(k) + " " + str(empty))
                    paniek = 1
                    
                #read from FIFO
                if (full > 2000):
                    ArrayCFloat = adw.GetFifo_Long(1, full)
                    ArrayFloat.extend(list(ArrayCFloat))
                    j += full
                    
        
        k = empty
        while k < InputSize:
            empty = adw.Fifo_Empty(2)
            full = adw.Fifo_Full(1)
            
            #check if read FIFO is empty enough
            if (adw.Fifo_Empty(1) < 5000):
                print("Houston we've got a read problem." + str(k) + " " + str(empty))
                paniek = 1
            
            #check if write FIFO is full enough
            if (adw.Fifo_Full(2) < 5000):
                print("Houston we've got a write problem." + str(k) + " " + str(empty))
                paniek = 1
            
            #read from FIFO
            if (full > 2000):
                ArrayCFloat = adw.GetFifo_Long(1, full)
                ArrayFloat.extend(list(ArrayCFloat))
            
            #write to FIFO     
            if (empty > 2000 and k + empty <= InputSize):
                adw.SetFifo_Long(2, list(InputBin[k:k+empty]), empty)
                k = k + empty
            elif (k + empty > InputSize):
                adw.SetFifo_Long(2, list(InputBin[k:InputSize]), InputSize - k)
                k = InputSize
                
        adw.Stop_Process(1)
        adw.Clear_Process(1)

except ADwinError as e:
    print('***', e)

#taxis = (3e3/3e8) * np.arange(len(ArrayFloat))
#ArrayFloat = [20 * (a - (65536 / 2)) / 65536 for a in ArrayFloat]
## plot data
plt.figure()
plt.plot(ArrayFloat,'o')
plt.plot(InputBin,'o')
plt.show()
