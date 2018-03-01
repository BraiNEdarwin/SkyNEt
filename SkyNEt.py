''''
Main handler of the SkyNEt platform
'''

import modules.ReservoirSparse as Reservoir

nodes = [1, 50, 1]  # [input, reservoir, output]
inputscaling = 0.2
spectralradius = 0.99
weightdensity = 1

res = Reservoir.Network(nodes, inputscaling, spectralradius, weightdensity)
