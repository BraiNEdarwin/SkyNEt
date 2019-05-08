#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:52:06 2019

@author: ljknoll

TODO:
    add legend option
    write documentation


Example:
    import time
    import numpy as np
    pb = PB()
    pb.add_subplot((0,0), (2,m),  ylim=(0,1), rowspan=2, xlabel='testx', title='testtitle', adaptive=True)
    pb.add_subplot((0,1), m, ylim=(0,1))
    pb.add_subplot((1,1), n, adaptive=True)    
    pb.finalize()
    
    t_start = time.time()
    for i in range(1, n):
        genome3 = np.random.rand(m)
        pb.update((0,1), genome3)
        if i%10==0:
            genomes = np.random.rand(2,i) + np.arange(i)/5
            pb.update((0,0), genomes)
            genome = np.random.rand(i)+np.arange(i)
            pb.update((1,1), genome)
    avg = (time.time() - t_start)/n
    print('Average time drawing one frame: %0.4f' %avg)


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class PlotBuilder:
    def __init__(self):
        assert matplotlib.get_backend() == 'Qt5Agg', "Error: In order to maximize figure, we need Qt5Agg backend. Try typing '%matplotlib' in ipython console"
        self.plots = {}
        self.data_size = {}
        self.adaptive = {}
    
    def add_subplot(self, pos, data_size, adaptive=False, **kwargs):
        """ 
        pos         tuple:  (y, x) position on figure grid
        data_size   int:    length of ydata in plot (ignored if adaptive)
                    tuple:  (nr_lines, n) plot number of lines each of size n
        adaptive    bool:   Whether axes sizes are adaptive to new data,
                             can be used together with ylim, but xlim argument is ignored.
        
        Warning: plotting takes some time (~50ms), if you want to speed things up, plot less often
        
        extra keyword arguments are used to set axis properties.
        e.g. if there is a keyword argument title = 'test title',
        then axes.set_title('test title') will be called on the subplot.
        See https://matplotlib.org/2.0.2/api/axes_api.html and search for set_ 
        """
        self.data_size[pos] = data_size
        self.adaptive[pos] = adaptive
        if adaptive:
            kwargs.pop('xlim', None)
        self.plots[pos] = kwargs

    
    def finalize(self):
        # determine size of grid (max_y, max_x)
        max_y = 0
        max_x = 0
        for pos in self.plots.keys():
            max_y = pos[0] if pos[0]>max_y else max_y
            max_x = pos[1] if pos[1]>max_x else max_x
        
        self.fig = plt.figure()
        self.axes = {}
        # loop through subplots we want to add and set properties of each subplot
        for pos, pltargs in self.plots.items():
            # add subplot in a grid (gridsize, position, colspan, rowspan)
            self.axes[pos] = plt.subplot2grid(
                    (max_y+1, max_x+1),
                    pos,
                    colspan=pltargs.pop('colspan', 1),
                    rowspan=pltargs.pop('rowspan', 1))
            self.axes[pos].set(**pltargs)
        
        # maximize figure
        figManager = self.fig.canvas.manager
        figManager.window.showMaximized()
        
        # It is necessary to draw the canvas at least once before draw_artist can be called
        self.fig.canvas.draw()
        
        # store lines and backgrounds in order to overwrite data later
        self.backgrounds = {}
        self.lines = {}
        for pos, args in self.plots.items():
            ax = self.axes[pos]
            size = self.data_size[pos]
            dummydata = np.zeros(size)
            # create and store lines such that their data can be overwritten later
            self.lines[pos] = ax.plot(np.arange(dummydata.shape[-1]), dummydata.T)
            self.backgrounds[pos] = self.fig.canvas.copy_from_bbox(ax.bbox)
        # use tight layout
        self.fig.tight_layout()
    
    def update(self, subplot, ydata):
        self.fig.canvas.restore_region(self.backgrounds[subplot])
        nr_points = ydata.shape[-1]
        ydata = ydata.reshape(-1,nr_points) # reshape to (nr_lines, nr_points) even for one dimension
        for i,line in enumerate(self.lines[subplot]):
            line.set_ydata(ydata[i])
            line.set_xdata(np.arange(nr_points))
        if self.adaptive[subplot]:
            self.axes[subplot].relim()
            self.axes[subplot].autoscale_view(True,True,True)
        self.fig.canvas.blit(self.axes[subplot].bbox)
        plt.pause(0.0000001)

