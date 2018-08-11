import numpy as np

a = np.random.rand(100,8,8)
b = np.round(a)

devlist=[3,5,7]

for e in range(len(devlist)):
    b[:,:,devlist[e]] = 0


for c in range(len(b)):
    duplicate = True
    templist = b[c]
    while(duplicate):
        stack = 0
        for d in range(len(b)):
            if np.array_equal(templist, b[d]):
                stack = stack + 1
        if stack == 1:
            b[c] = b[c]
            duplicate = False
        if stack > 1:
            newlist = np.random.rand(8,8)
            templist = np.round(newlist)
            for e in range(len(devlist)):
                templist[:,:,devlist[e]] = 0