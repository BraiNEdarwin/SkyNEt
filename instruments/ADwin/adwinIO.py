'''
This module provides an input/output function for communicating with the
ADwin.
'''
from instruments.ADwin.adwin import ADwin, ADwinError
import sys
import os
import numpy as np

DEVICENUMBER = 1
RAISE_EXCEPTIONS = 1
PROCESSORTYPE = '11'
PROCESS = 'ADbasic_1Read_1Write.TB1'
FifoSize = 40003



def initInstrument():
    adw = ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
    return adw


def IO(adw, x, Fs):
    x = np.asarray(x)  #convert x to numpy array
    x = (x + 10) / 20 * 65536
    x = x.astype(int)
    ArrayFloat = []
    InputBin = x.copy() # convert float array to integer values
    InputSize = len(x)
    print(InputBin)
    # trail zeros if data is shorter than buffer
    try:
        #adw = ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
        if os.name == 'posix':
            adw.Boot(str('adwin' + PROCESSORTYPE + '.btl'))
        else:
            adw.Boot('C:\\ADwin\\ADwin' + PROCESSORTYPE + '.btl')
        #proc = os.path.abspath(os.path.dirname(sys.argv[0])) + os.sep + 'intstruments' + os.sep + 'ADwin' + os.sep + PROCESS
        adw.Load_Process('C:\\Users\\ursaminor\\Documents\\GitHub\\SkyNEt\\instruments\\ADwin\\ADbasic_1Read_1Write.TB1')
        adw.Set_Processdelay(1, int(300e6 / Fs))  # delay in clock cycles

        # fill the write FIFO
        adw.Fifo_Clear(2)
        empty = adw.Fifo_Empty(2)
        if (InputSize < empty):
            adw.SetFifo_Long(2, list(InputBin), len(InputBin))
        else:
            adw.SetFifo_Long(2, list(InputBin[:empty]), empty)


        adw.Start_Process(1)
        # time.sleep(0.3)  # Give init time

        if (InputSize < FifoSize):
            j = 0  # counts amount of read values
            while(j < InputSize):
                full = adw.Fifo_Full(1)

                # check if read FIFO is empty enough
                if (adw.Fifo_Empty(1) < 5000):
                    print("Houston we've got a read problem." +
                          str(j) + " " + str(empty))
                    paniek = 1

                # read from FIFO
                if (full > 2000):
                    ArrayCFloat = adw.GetFifo_Long(1, full)
                    ArrayFloat.extend(list(ArrayCFloat))
                    j += full

        else:
            k = 0  # counts no. of written datapoints
            writeFinished = False
            j = 0  # counts no. of read datapoints
            k += empty
            while j < InputSize:
                empty = adw.Fifo_Empty(2)
                full = adw.Fifo_Full(1)

                # check if read FIFO is empty enough
                if (adw.Fifo_Empty(1) < 5000):
                    print("Houston we've got a read problem." +
                          str(k) + " " + str(empty))
                    paniek = 1

                # check if write FIFO is full enough
                if (adw.Fifo_Full(2) < 5000 and not writeFinished):
                    print("Houston we've got a write problem." +
                          str(k) + " " + str(empty))
                    paniek = 1

                # read from FIFO
                if (full > 2000):
                    ArrayCFloat = adw.GetFifo_Long(1, full)
                    ArrayFloat.extend(list(ArrayCFloat))
                    j += full

                # write to FIFO
                if (empty > 2000 and k + empty <= InputSize): 
                    adw.SetFifo_Long(2, list(InputBin[k:k + empty]), empty)
                    k = k + empty
                elif (k + empty > InputSize and not writeFinished):
                    adw.SetFifo_Long(2, list(InputBin[k:InputSize]), InputSize - k)
                    k = InputSize
                    writeFinished = True #write fifo is now filled with last of InputBin

        adw.Stop_Process(1)
        adw.Clear_Process(1)

    except ADwinError as e:
        print('***', e)
    # convert int to float
    ArrayFloat = [20 * (a - (65536 / 2)) / 65536 for a in ArrayFloat]

    return ArrayFloat[:InputSize] # trim off excess datapoints
