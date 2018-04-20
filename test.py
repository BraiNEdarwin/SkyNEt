


import numpy as np
import itertools as iter

a = [-1,0,1]
b = iter.product(a, repeat=5)
c = list(b)

generange = [[-0.57084235, -0.57084235], [0.32070898, 0.32070898], [-0.08166972, -0.08166972], [-1.83939985, -1.83939985], [-0.34799499, -0.34799499], [0.5, 0.5]]

for i in range(5):
	generange[i][0] = c[5][i]
	generange[i][1] = c[5][i]

print(generange)

# for n in range(len(b)-1)
# 	generange = 