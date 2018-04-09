import modules.PostProcess as PostProcess
import modules.GenerateInput as GenerateInput
import numpy as np
import matplotlib.pyplot as plt


t, x, y, W = GenerateInput.softwareInput(['bl','AND'], 1000, 5, 10)

target_weighed = PostProcess.fitnessEvolution(x, y, W)

plt.plot(target_weighed)
plt.show()
