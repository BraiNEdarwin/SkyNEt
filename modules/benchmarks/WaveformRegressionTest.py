import WaveformRegression as wr
import matplotlib.pyplot as plt

Fs = 1e3


[t, x] = wr.sineWave(Fs)


plt.figure()
plt.plot(t,x)
plt.show()
