'''
A simple driver for the Keithley2000.
For now it only supports reading N voltages at a certain sample frequency
'''

import visa
import time

def initInstrument(address):
  '''
  Address should be a valid GPIB address, such as 'GPIB0::17::INSTR'
  '''
  rm = visa.ResourceManager()
  keithley = rm.open_resource(address)
  keithley.write('*rst; status:preset; *cls')
  return keithley

def readValues(keithley, N, Fs):
  '''
  Read N values at sample frequency Fs
  This has been adapted from the example found here:
  https://pyvisa.readthedocs.io/en/stable/example.html
  The function is still very slow, but should do the job
  '''
  # Prepare Keithley
  keithley.write('status:measurement:enable 512; *sre 1')
  keithley.write(f'sample:count {N}')
  keithley.write('trigger:source bus')
  keithley.write(f'trigger:delay {1/Fs}')
  keithley.write(f'trace:points {N}')
  keithley.write('trace:feed sense1; feed:control next')

  keithley.write('initiate')

  keithley.assert_trigger()
  keithley.wait_for_srq()

  voltages = keithley.query_ascii_values('trace:data?')

  keithley.query('status:measurement?')
  keithley.write('trace:clear; feed:control next')

  return voltages



