import BooleanLogic
import matplotlib.pyplot as plt
t, x, y = BooleanLogic.InputSignals(1000)
t, x_or = BooleanLogic.NOR(1000)

# plt.plot(t, x)
# plt.plot(t, y)
plt.plot(t, x_or)
plt.show()