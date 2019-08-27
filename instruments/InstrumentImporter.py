# This simple file is a wrapper for 
# importing all measurement equipment available in the lab.
# It also (importantly!) sets up a reset function that is executed
# at ctrl-C

from SkyNEt.instruments.ADwin import adwinIO
from SkyNEt.instruments.niDAQ import nidaqIO
from SkyNEt.instruments.DAC import IVVIrack
#import serial
import signal
import sys

def reset(signum, frame, exit=True):
        '''
        This functions performs the following reset tasks:
        - Set IVVI rack DACs to zero
        - Apply zero signal to the NI daq
        - Apply zero signal to the ADwin
        - exits script if exit=True (default)
        '''
        ivvi_found = False
        reset_ivvi = False
        for i in range(5):
            try:
                # Check if comport has ivvi by reading serial response
                #ser = serial.Serial(port=f'COM{i+1}', baudrate=115200, timeout=1)
                #ser.write(bytes([2, 4]))  # This queries the version

                # The IVVI sends b'\x02\xf0' back
                #answer = ser.read(2)
                #if(answer == b'\x02\xf0'):
                #    ivvi_found = True
                #ser.close()

                # Reset the ivvi dacs

                ivviReset = IVVIrack.initInstrument(name='ivviReset' + str(i+1), comport='COM' + str(i+1))
                ivviReset.set_dacs_zero()
                print('ivvi DACs set to zero')
                reset_ivvi = True
            except: pass
        if not reset_ivvi:    
            print('ivvi was not initialized, so also not reset')

        try:
            nidaqIO.reset_device()
            print('nidaq has been reset')
        except:
            print('nidaq not connected to PC, so also not reset')

        try:
            adw = adwinIO.initInstrument()
            adwinIO.reset(adw)

            print('adwin has been reset')
        except:
            print('adwin was not initialized, so also not reset')

        if(exit==True):
            sys.exit()
	
# Set up reset call at ctrl-C
signal.signal(signal.SIGINT, reset)

