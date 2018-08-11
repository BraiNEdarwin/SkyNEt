from instrument import Keith2400
import numpy as np
import time

keithley = Keith2400.Keithley_2400('keithley', 'GPIB0::11')

keithley.compliancei.set(100E-6)

keithley.volt.set(1)

test = np.zeros(75)
print(test)
x = 0
while(x < 60):
	keithley.volt.set(x*0.01)
	time.sleep(1)
	current = keithley.curr.get()
	print(current)
	test[x] = current
	x= x+1

