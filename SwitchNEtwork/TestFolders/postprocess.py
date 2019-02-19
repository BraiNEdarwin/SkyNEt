#suppose we have list of 10 elements, representing the fitness score of the 10 switch configurations
#Bigmommy has an associated fitness score
import numpy as np
import operator

genomes = 10
genes = 8
bigdaddy = np.random.rand(genomes,genes,genes)
bigmommy = np.around(bigdaddy)
#print(bigmommy)

examplelist = [1, 56, 21, -4, 97, 90, 179, 43, 21, 65]
print(examplelist)

winner = examplelist.index(max(examplelist))
print (winner)

count = 0
m1 = m2 = float('-inf')
for x in examplelist:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
if count>=2:
    	silver = m2

secondwinner = examplelist.index(silver)
print(secondwinner)

#identify the winner and the second

print('The winner is:')
print(bigmommy[winner])

newGenConfigs = np.zeros((genomes, genes, genes))

#Winner remains = 1
newGenConfigs[0] = bigmommy[winner]

#Winner with slight modification = 4
for i in range(1, 8):
    templist = bigmommy[winner]
    for j in range(genes):
        for k in range(genes):
            if(np.random.rand() < 0.1):
            	if templist[j, k] == 1:
            		templist[j, k] = 0
            	elif templist[j, k] == 0:
                	templist[j, k] = 1
    newGenConfigs[i] = templist

#Is it possible to utilize the second winner for the breeding?

#complete random = 2
for i in range(8,10):
    templist = np.random.rand(genes,genes)
    newGenConfigs[i] = np.around(templist)



