import modules.PostProcess as PostProcess
import modules.GenerateInput as GenerateInput
import numpy as np
import matplotlib.pyplot as plt


t, x, y, W = GenerateInput.softwareInput(['bl','AND'], 1000, 5, 10)
par = [1, 1, 1, 0.1]
test_data = np.empty(530)
test_data[0:125] = 0
test_data[125:135] = -1
test_data[135:260] = 0.5
test_data[260:270] = -1
test_data[270:395] = 0
test_data[395:405] = -1
test_data[405:530] = 1
noise = np.random.rand(530) * 0.1
test_data = test_data + noise
F, m, c = PostProcess.fitnessEvolution(test_data, x, W, par)

plt.plot(x)
plt.plot((test_data - c) / m)
#plt.plot(min0 * np.ones(530),'r--')
#plt.plot(max0 * np.ones(530),'r--')
#plt.plot(min1 * np.ones(530),'b--')
#plt.plot(max1 * np.ones(530),'b--')
plt.show()
