'''
This module provides an input/output function for communicating with the
ADwin.
There is also a separate function that allows you to use the ADwin
output ports as 'static' control voltages. This function may need
some refinement in the future.
'''
from SkyNEt.instruments.ADwin.adwin import ADwin, ADwinError
import sys
import os
import numpy as np
import time

DEVICENUMBER = 1
RAISE_EXCEPTIONS = 1
PROCESSORTYPE = '11'
PROCESS = 'ADbasic_1Read_1Write.TB1'
FifoSize = 40003

def initInstrument():
    adw = ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
    return adw

def FloatToLong(x):
    '''
    Converts float values to integers,
    more specifically this function maps [-10, 10] -> [0, 65536]
    '''
    for i in range(len(x)):
        x[i] = int((x[i] + 10) / 20 * 65536)
    return x

def LongToFloat(x, Vmax):
    '''
    Converts integer values to floats,
    more specifically this function maps [0, 65536] -> [-Vmax, Vmax]
    '''
    for i in range(len(x)):
        x[i] = 2*Vmax/65536 * x[i] - Vmax
    return x


def IO(adw, input, Fs, inputPorts = [1, 0, 0, 0, 0, 0, 0]):
    '''
    This function will write each row of array inputs on a separate
    analog output of the ADwin at the specified sample frequency Fs.
    inputs must be a numpy array, where each row (!) corresponds to
    data presented on one output port.
    inputPorts in an optional argument that allows you to specify which
    analog input ports you wish to measure. Note that this function
    only reads on the EVEN ADCs (so 2, 4, ...).

    Useful information if you want to understand this function:
    Please look at the ADbasic file while reading this file. 
    Write FIFOs: 1 - 4
    Read FIFOs: 5 - 12
    '''
    # Input preparation
    inputs = input.copy()
    InputSize = inputs.shape[1]
    for i in range(inputs.shape[0]):
        inputs[i, :] = FloatToLong(list(inputs[i, :]))
    x = np.zeros((8, InputSize), dtype = int)
    x[:inputs.shape[0], :] = inputs
    outputs = [[], [], [], [], [], [], [], []]  # Eight empty output lists

    try:
        if os.name == 'posix':
            adw.Boot(str('adwin' + PROCESSORTYPE + '.btl'))
        else:
            adw.Boot('C:\\ADwin\\ADwin' + PROCESSORTYPE + '.btl')
        adw.Load_Process('C:\\Users\\PNPNteam\\Documents\\GitHub\\SkyNEt\\instruments\\ADwin\\ADbasic_8Read_4Write.TB1')
        adw.Set_Processdelay(1, int(300e6 / Fs))  # delay in clock cycles

        adw.Start_Process(1)

        # Clear all FIFOs
        for i in range(1, 13):
            adw.Fifo_Clear(i)
        
        # Fill write FIFOs before start of reading/writing
        if(FifoSize <= InputSize):
            for i in range(1, 5):
                fillSize = adw.Fifo_Empty(i)
                adw.SetFifo_Long(i, list(x[i-1,:fillSize]), fillSize)
                written = fillSize
        else:
            for i in range(1, 5):
                adw.SetFifo_Long(i, list(x[i-1, :]), InputSize)
                written = InputSize

        # Start reading/writing FIFO's when Par80 == 1
        adw.Set_Par(80, 1)
        read = -1  # Read additional datapoint, because write lags behind

        while(read < InputSize):
            empty = adw.Fifo_Empty(1)
            full = adw.Fifo_Full(5)

            # Read values if read FIFOs are full enough
            if(full > 2000):
                for i in range(5, 13):          # Read ports are 5 to 12
                    y = adw.GetFifo_Long(i, 2000) 
                    outputs[i-5] += LongToFloat(list(y), 5)
                read += 2000

            # Write values if write FIFOs are empty enough
            if(written < InputSize):
                if(empty > 2000 and written+2000 <= InputSize):
                    for i in range(1, 5):
                        adw.SetFifo_Long(i, list(x[i-1, written:written + 2000]), 2000)
                    written += 2000
                elif(empty > 2000):
                    for i in range(1, 5):
                        adw.SetFifo_Long(i, list(x[i-1, written:]), InputSize-written)
                    written = InputSize

        adw.Stop_Process(1)
        adw.Clear_Process(1)

    except ADwinError as e:
        print('***', e)

    # Prepare outputArray
    outputArray = np.zeros((sum(inputPorts), InputSize))
    outputs = np.array(outputs)
    i = 0
    for index, val in enumerate(inputPorts):
        if(val):
            outputArray[i] = outputs[index, 1:InputSize + 1]
            i += 1

    return outputArray


def setControlVoltages(adw, x, Fs):
    '''
    x is a list of 4 values with desired control voltages in V.
    '''
    x = np.asarray(x)  #convert x to numpy array
    x = (x + 10) / 20 * 65536
    x = x.astype(int)
    
    x = np.tile(x, (FifoSize, 1))

    InputBin = x.copy() # convert float array to integer values
    InputSize = len(x)
    try:
        #adw = ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
        if os.name == 'posix':
            adw.Boot(str('adwin' + PROCESSORTYPE + '.btl'))
        else:
            adw.Boot('C:\\ADwin\\ADwin' + PROCESSORTYPE + '.btl')
        #proc = os.path.abspath(os.path.dirname(sys.argv[0])) + os.sep + 'intstruments' + os.sep + 'ADwin' + os.sep + PROCESS
        adw.Load_Process('C:\\Users\\PNPNteam\\Documents\\GitHub\\SkyNEt\\instruments\\ADwin\\ADbasic_8write.TB1')
        adw.Set_Processdelay(1, int(300e6 / Fs))  # delay in clock cycles

        # fill the write FIFO
        print(x[:, 1])
        for i in range(4):
            adw.Fifo_Clear(i + 1)
            adw.SetFifo_Long(i+1, list(x[:, i]), FifoSize)

        adw.Start_Process(1)

    except ADwinError as e:
        print('***', e)

