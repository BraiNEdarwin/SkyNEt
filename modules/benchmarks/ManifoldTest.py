import Manifold
import matplotlib.pyplot as plt
t, x, y, z, v, w= Manifold.InputSignals(1000)
# t, x_or = BooleanLogic.NOR(1000)

# plt.plot(t, x)
# plt.plot(t, y)
plt.plot(t, w)
plt.show()