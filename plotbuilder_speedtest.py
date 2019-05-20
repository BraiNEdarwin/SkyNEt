from modules.PlotBuilder import PlotBuilder
import time
import numpy as np
m,n = 10,100
pb = PlotBuilder()
pb.add_subplot('big_plot', (0,0), (2,0),  ylim=(0,1), adaptive=True, rowspan=2, title='adaptive X, static Y', xlabel='generations')
pb.add_subplot('static', (0,1), m, ylim=(0,1), title='static axis are quicker', xlabel='genes')
pb.add_subplot('adaptive', (1,1), n, adaptive=True, title='both axes are adaptive', xlabel='iteration')
pb.finalize()

t_start = time.time()
genomes = np.zeros((2,n))
for i in range(1, n):
    genome3 = np.random.rand(m)
    pb.update('static', genome3)
    if i%10==0:
        genomes[:,i//10] = np.random.rand(2)
        pb.update('big_plot', genomes[:, :i//10+1])
        genome = np.random.rand(i)*n/m+np.arange(i)
        pb.update('adaptive', genome)
avg = (time.time() - t_start)/n
print('Average time drawing one frame: %0.4f' %avg)