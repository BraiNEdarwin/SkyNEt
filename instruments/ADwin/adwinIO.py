'''
This module provides an input/output function for communicating with the
ADwin.
There is also a separate function that allows you to use the ADwin
output ports as 'static' control voltages. This function may need
some refinement in the future.
'''
from SkyNEt.instruments.ADwin.adwin import ADwin, ADwinError
from SkyNEt.instruments import InstrumentImporter
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

DEVICENUMBER = 1
RAISE_EXCEPTIONS = 1
PROCESSORTYPE = '11'
PROCESS = 'ADbasic_1Read_1Write.TB1'
FifoSize = 40003

def initInstrument():
    adw = ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
    return adw

def FloatToLong(x, Vmax):
    '''
    Converts float values to integers,
    more specifically this function maps [-10, 10) -> [0, 65536)
    '''
    for i in range(len(x)):
        x[i] = int(np.round((x[i] + Vmax) / (2*Vmax / 65535)))
    return x

def LongToFloat(x, Vmax):
    '''
    Converts integer values to floats,
    more specifically this function maps [0, 65536) -> [-Vmax, Vmax)
    '''
    if isinstance(x, int):
        x = 2*Vmax/65536 * x - Vmax
    else:
        for i in range(len(x)):
            x[i] = 2*Vmax/65536 * x[i] - Vmax
    return x


def IO(adw, Input, Fs, inputPorts = [1, 0, 0, 0, 0, 0, 0], highRange = False):
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

    Inputs arguments
    ----------------
    adw: adwin
    Input: N x M array, N output ports, M datapoints
    Fs: sample frequency
    inputPorts**: binary list containing ones for the used input ports

    Returns
    -------
    P x M output array, P input ports, M datapoints
    '''
    # Sanity check on input voltages
    if not highRange:
        if max(abs(y) > 2):  
            print('WARNING: input voltages exceed threshold of 2V: highest absolute voltage is ' + str(max(abs(y))))
            print('If you want to use high range voltages, set highRange to True.')
            print('Aborting measurement...')
            InstrumentImporter.reset(0, 0)
            exit() 

    # Input preparation
    if len(Input.shape) == 1:
        Input = Input[np.newaxis,:]

    inputs = Input.copy()
    InputSize = inputs.shape[1]
    x = np.zeros((8, InputSize), dtype = int)

    # Transform all inputs to Long:
    for i in range(x.shape[0]):
        if i < inputs.shape[0] :
            x[i, :] = FloatToLong(list(inputs[i, :]), 10)
        else:
            x[i, :] = FloatToLong(list(x[i, :]), 10)
    outputs = [[], [], [], [], [], [], [], []]  # Eight empty output lists
    lastWrite = False

    try:
        if os.name == 'posix':
            adw.Boot(str('adwin' + PROCESSORTYPE + '.btl'))
        else:
            adw.Boot('C:\\ADwin\\ADwin' + PROCESSORTYPE + '.btl')
        adw.Load_Process('C:\\Users\\Darwin\\Documents\\GitHub\\SkyNEt\\instruments\\ADwin\\ADbasic_8Read_4Write.TB1')
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
                lastWrite = True



        # Notices the ADbasic how much datapoints to
        adw.Set_Par(79, InputSize)
        # Start reading/writing FIFO's when Par80 == 1
        adw.Set_Par(80, 1)
        read = -1  # Read additional datapoint, because write lags behind

        if (lastWrite):
            time.sleep((InputSize - read)/Fs) # If the last values are put in the write memory, wait a bit until they are written

        while(read < InputSize):
            empty = adw.Fifo_Empty(1)
            full = adw.Fifo_Full(5)

            # Read values if read FIFOs are full enough
            if(full > 2000 and not lastWrite):
                for i in range(5, 13):          # Read ports are 5 to 12
                    y = adw.GetFifo_Long(i, 2000) 
                    outputs[i-5] += LongToFloat(list(y), 10) 
                read += 2000

            elif(lastWrite):
                for i in range(5, 13):          # Read ports are 5 to 12
                    y = adw.GetFifo_Long(i, full) 
                    outputs[i-5] += LongToFloat(list(y), 10) 
                read += full

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
                    lastWrite = True
                    time.sleep((InputSize - read)/Fs) # If the last values are put in the write memory, wait a bit until they are written

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
            outputArray[i] = outputs[index, 1:InputSize+1]
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
        adw.Load_Process('C:\\Users\\Darwin\\Documents\\GitHub\\SkyNEt\\instruments\\ADwin\\ADbasic_8write.TB1')
        adw.Set_Processdelay(1, int(300e6 / Fs))  # delay in clock cycles

        # fill the write FIFO
        print(x[:, 1])
        for i in range(4):
            adw.Fifo_Clear(i + 1)
            adw.SetFifo_Long(i+1, list(x[:, i]), FifoSize)

        adw.Start_Process(1)

    except ADwinError as e:
        print('***', e)


def reset(adw):
    '''
    Resets the 4 DACs to 0V at a speed of 0.5V/s
    '''
    Fs = 1000
    stepsize = 0.0005 # in units of V
    DAC_values = []
    DAC_diff = []
    no_steps = []
    
    for i in range(1,5):
        DAC_values += [LongToFloat(adw.Get_Par(i), 10)] 
        no_steps += [int(abs(DAC_values[i-1]/stepsize))]

    reset_inputs = np.zeros((4, max(no_steps)))
    for i in range(reset_inputs.shape[0]):
        reset_inputs[i,:no_steps[i]] = np.linspace(DAC_values[i], 0, no_steps[i], endpoint = True)

    resetData = IO(adw, reset_inputs, Fs)


