import numpy as np
Output = np.random.rand(8,8)
F = 0
#Tolerance. if set 0.5, it considers any output that has more than 50% of the highest current as "non-distinguishable"
threshold = 0.5
#Criteria 1
TransOutput = np.transpose(Output)
for a in range(len(Output)):
    count = 0
    tempout = TransOutput[a]
    maxi = max(tempout)
    for b in range(len(Output[a])):

        #If the read current is higher than the threshold, add 1 to the count
        if (TransOutput[a,b]/maxi) > threshold:
            count = count + 1
            #if only one output was HIGH for the given input, that's success!
    if count == 1:
        F = F + 3
			#if more than 1 output was HIGH for the given input, we give -1 for the number of outputs that were HIGH
    elif count > 1:
        F = F+ -1*count
			#if no output was HIGH for a particular input, we either have to lower the threshold, or just punish the fitness score
    elif count == 0:
        F = F - 10
    print(F)