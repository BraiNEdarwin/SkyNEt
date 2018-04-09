import PostProcess
import GenerateInput
import numpy as np
import matplotlib.pyplot as plt


t, x, y, W = GenerateInput.softwareInput('bl', 1000, 5, 10)


plt.plot(t,x)
plt.plot(t,y)
plt.plot(t,W)
plt.show()
