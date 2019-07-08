import itertools
import numpy as np

a = [0, 1]
inp = np.array(list(itertools.product(*[a,a,a,a])))
print(inp)