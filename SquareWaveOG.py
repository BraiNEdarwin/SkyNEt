import matplotlib.pyplot as plt 
import numpy as np 
import config_SWOG as config

# Load the information from the config class.
config = config.experiment_config()
Input = config.SquareWave( config.v_high, config.v_low, config.n_points)
x = np.arange(len(Input))

# Plot the Square wave

plt.figure()
plt.plot(x, Input)
plt.show()