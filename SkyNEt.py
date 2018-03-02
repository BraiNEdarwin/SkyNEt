''''
Main handler of the SkyNEt platform
'''
# Import packages
import modules.ReservoirSparse as Reservoir
import modules.PlotBuilder as PlotBuilder
import modules.GenerateInput as GenerateInput
import math
# temporary imports
import numpy as np

# Read config.txt file
exec(open("config.txt").read())

# Init software reservoir
print("Initializing the reservoir...")
res = Reservoir.Network(nodes, inputscaling, spectralradius, weightdensity)

# Obtain benchmark input
inp = GenerateInput.softwareInput(benchmark, SampleFreq)

print("Feeding the input signal...")
printcounter = 0
for i in range(len(inp)):
    if (printcounter == math.floor(len(inp) / 10)):
        print('%d%% completed' % ((i / len(inp)) * 100), end='\r')
        printcounter = 0
    else:
        printcounter += 1
    res.update_reservoir(inp[i])

# temporary plot
y = np.empty((len(x), 5))
for i in range(5):
    y[:,i] = res.collect_state[:, i]

PlotBuilder.genericPlot(x, y, 'Time (A.U.)', 'Output (A.U.)', 'Example reservoir states')