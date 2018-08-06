'''
This module provides an input/output function for communicating with the
NI USB 6216. (Note that nidaqmx has to be installed seperately first.)
For now it is one output, one input.
'''
import nidaqmx
import numpy as np

def IO(y, Fs):
    N = len(y)
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task: 
        #define ao/ai channels
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao0', 'ao0', -10, 10)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')

        #configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan = N)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan = N)

        #output triggers on the read operation
        output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/ai/StartTrigger')

        #fill the output buffer
        output_task.write(y, auto_start=False)

        #start tasks
        output_task.start()
        input_task.start()
        
        #read data
        data = input_task.read(N)

        #stop and close the tasks
        input_task.stop()
        #input_task.close()
        output_task.stop()
        #output_task.close()
		
    return data


def IO_2D(x, Fs):
    N = x.shape[1]
    np.append(x, [[0], [0]], axis=1)  #finish by setting dacs to 0
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task: 
        #define ao/ai channels
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao0', 'ao0', -10, 10)
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao1', 'ao1', -10, 10)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')

        #configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan = N + 1)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan = N + 1)

        #output triggers on the read operation
        output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/ai/StartTrigger')

        #fill the output buffer
        output_task.write(x, auto_start=False)

        #start tasks
        output_task.start()
        input_task.start()
        
        #read data
        read_data = input_task.read(N + 1)
        data = read_data[1:] #trim off the first datapoint, read lags one sample behind write

        #stop and close the tasks
        input_task.stop()
        #input_task.close()
        output_task.stop()
        #output_task.close()
        
    return data

    def IO_2D2I(x, Fs):
    N = x.shape[1]
    np.append(x, [[0], [0]], axis=1)  #finish by setting dacs to 0
    with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task: 
        #define ao/ai channels
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao0', 'ao0', -10, 10)
        output_task.ao_channels.add_ao_voltage_chan('Dev1/ao1', 'ao1', -10, 10)
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')
        input_task.ai_channels.add_ai_voltage_chan('Dev1/ai1')

        #configure sample rate and set acquisition mode to finite
        output_task.timing.cfg_samp_clk_timing(Fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan = N + 1)
        input_task.timing.cfg_samp_clk_timing(Fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan = N + 1)

        #output triggers on the read operation
        output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/ai/StartTrigger')

        #fill the output buffer
        output_task.write(x, auto_start=False)

        #start tasks
        output_task.start()
        input_task.start()
        
        #read data
        read_data = input_task.read(N + 1)
        data = read_data[1:] #trim off the first datapoint, read lags one sample behind write

        #stop and close the tasks
        input_task.stop()
        #input_task.close()
        output_task.stop()
        #output_task.close()
        
    return data