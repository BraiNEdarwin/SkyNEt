import WaveformRegression as wr
import matplotlib.pyplot as plt

Fs = 1e3
npoints = 1e3
frequency = 10e3
amplitude = 1

x = wr.SineWave(Fs, npoints)
y = wr.SquareWave(Fs, npoints)
z = wr.SawTooth(Fs, npoints)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.show()
