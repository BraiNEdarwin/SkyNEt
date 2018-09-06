import config_Boolean as config
import matplotlib.pyplot as plt
import numpy as np

config = config.experiment_config()
a = config.InputGen()[1:3]
print(config.InputGen()[1:2])

plt.plot(config.InputGen()[0],config.InputGen()[1:3])

