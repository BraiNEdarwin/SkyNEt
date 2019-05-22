#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:52:06 2019

@author: ljknoll

The class PlotBuilder can be used to initialize a figure, then only update it when you have new data.
For this reason, animations cannot be used because these are based on timers.
Figure is not blocking and can be closed whenever you want

requires Qt5 backend and at least matplotlib version 3 (is checked for)


Matplotlib overview of Objects:
    stolen from: https://stackoverflow.com/questions/14844223/python-matplotlib-blit-to-axes-or-sides-of-the-figure
    
    Figure
        Axes (0-many) (An axes is basically a subplot)
            Axis (usually two) (x-axis and y-axis)
                ticks
                ticklabels
                axis label
             background patch
             title, if present
             anything you've plotted, e.g. Line2D's

not supported yet:
    set only lower or upper limit

Example:
    import time
    import numpy as np
    from SkyNEt.modules.PlotBuilder import PlotBuilder
    m,n = 10,100
    pb = PlotBuilder()
    pb.add_subplot('big_plot', (0,0), (2,0),  ylim=(0,1), adaptive='x', rowspan=2, title='adaptive X, static Y', xlabel='generations', legend=['one', 'two'])
    pb.add_subplot('static', (0,1), m, ylim=(0,1), title='static axis are much quicker', xlabel='genes')
    pb.add_subplot('adaptive', (1,1), n, adaptive='both', title='both axes are adaptive', xlabel='iteration')
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

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class PlotBuilder:
    """
    Initializes a figure to plot live data.
    
    optional arguments:
        style       (str)   one of the styles to use for this figure, for list of available styles type matplotlib.style.available
        maximize    (bool)  whether to maximize the figure or not requires Qt backend
        lineprops   (dict)  lineproperties to set to all lines in the graph
    
    """
    def __init__(self, style='ggplot', maximize=True, lineprops = {}):
        assert int(matplotlib.__version__[0])>=3, 'matplotlib version (%s) too old, please update matplotlib to at least 3.0.0' % matplotlib.__version__
        
        assert matplotlib.get_backend() == 'Qt5Agg', "\x1b[33mWARN: In order to maximize figure, we need Qt backend. Try typing '%matplotlib qt' in your ipython console.\x1b[0m"
        
        if style == 'default' or style in plt.style.available:
            self.style = style
        else:
            # see https://matplotlib.org/api/style_api.html for details
            print("\x1b[33mWARN: style %s is not available so using default, you can choose from: " % style)
            print(plt.style.available)
            self.style = 'default'
        
        self.maximize = maximize
        self.plots = {}
        self.data_size = {}
        self.adaptive = {}
        self.legends = {}
        self.names = {}
        self.lineprops = {}
        self.default_lineprops = lineprops
    
    def add_subplot(self, name, pos, data_size, adaptive=False, legend=None, lineprop={}, **kwargs):
        """
        Adds a subplot to the figure at grid position pos
        
        name        string: name handle which can be used to update the figure
        pos         tuple:  (y, x) position on figure grid
        data_size   int:    length of ydata in plot (ignored if adaptive)
                    tuple:  (nr_lines, n) plot number of lines each of size n
        adaptive    string: Whether axes are adaptive to new data 'x', 'y' or 'both'. xlim or ylim are overwritten by adaptive.
            WARNING: adaptive axes are slow (~50ms per subplot in your figure) and should not be used in between measurements, use static plots or update less often
        legend      list:   List of strings to be put in the legend
        lineprop    dict:   dictionary of line properties, example: {'marker':'x'}, see
                                https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        
        
        extra keyword arguments are used to set axes properties.
        e.g. if there is a keyword argument title = 'test title',
        then axes.set_title('test title') will be called on the subplot.
        
        See link:
        https://matplotlib.org/api/axes_api.html#the-axes-class
        and search '.set_' for all available properties
        
        """
        self.data_size[pos] = data_size
        self.adaptive[pos] = adaptive
        self.legends[pos] = legend
        self.plots[pos] = kwargs
        self.names[name] = pos
        self.lineprops[pos] = lineprop

    
    def finalize(self):
        """
        Actually builds and shows the figure, only call this after all subplots have been added.
        Caches and stores everything in such a way that updating subplots is as fast as possible.
        """
        # determine size of grid (max_y, max_x)
        max_y = 0
        max_x = 0
        for pos in self.plots.keys():
            max_y = pos[0] if pos[0]>max_y else max_y
            max_x = pos[1] if pos[1]>max_x else max_x
        
        # use temporary style
        with plt.style.context((self.style)):
            self.fig = plt.figure()
            self.fig.show()
            
            if self.maximize:
                # maximize figure, works only on Qt backends
                figManager = self.fig.canvas.manager
                figManager.window.showMaximized()
            gs = self.fig.add_gridspec(max_y+1, max_x+1)
            self.axes = {}
            # loop through subplots we want to add and set properties of each subplot
            for pos, pltargs in self.plots.items():
                rows = pltargs.pop('rowspan',1)
                cols= pltargs.pop('colspan',1)
                gs_temp = gs[pos[0]:pos[0]+rows, pos[1]:pos[1]+cols]
                self.axes[pos] = self.fig.add_subplot(gs_temp, **pltargs)
                self.axes[pos].grid(True, which='major')

            # use tight layout
            self.fig.set_tight_layout(True)
            # It is necessary to draw the canvas at least once before draw_artist can be called
            self.fig.canvas.draw()

            # wait for a bit so we avoid weird artifacts when plotting fast, 
            # probably because background would not be done drawing yet
            self.fig.canvas.start_event_loop(0.5)
            self.backgrounds = {}
            for pos in self.plots.keys():
                self.backgrounds[pos] = self.fig.canvas.copy_from_bbox(self.axes[pos].bbox)

            # plot and store lines
            self.lines = {}
            for pos, args in self.plots.items():
                ax = self.axes[pos]
                size = self.data_size[pos]
                dummydata = np.zeros(size)
                # create and store lines such that their data can be overwritten later
                self.lines[pos] = ax.plot(np.arange(dummydata.shape[-1]), 
                                          dummydata.T,
                                          # overwrite default line properties
                                          **{**self.default_lineprops, **self.lineprops[pos]})
                if self.legends[pos] is not None:
                    ax.legend(self.lines[pos], self.legends[pos], loc='upper left')

    def update(self, name, ydata):
        """
        Updates a single subplot named (name) with data (ydata)
        WARNING: updates the complete figure (all subplots) when adaptive is true
        """
        # grab subplot
        subplot = self.names[name]
        # prepare new data
        nr_points = ydata.shape[-1]
        xdata = np.arange(nr_points)
        ydata = ydata.reshape(-1,nr_points) # reshape to (nr_lines, nr_points), also for a single line
        # grab axes of subplot (not axis!)
        axs = self.axes[subplot]
        # restore subplot to original state
        axs.draw_artist(axs.patch)
        self.fig.canvas.restore_region(self.backgrounds[subplot])
        # update lines
        for i,line in enumerate(self.lines[subplot]):
            line.set_ydata(ydata[i])
            line.set_xdata(xdata)
            axs.draw_artist(line)
        if self.adaptive[subplot]:
            # if adaptive is x, y or both, redraw complete canvas including xticks, labels etc
            axs.relim()
            axs.autoscale(enable=True, axis=self.adaptive[subplot])
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.0001)
        else:
            # otherwise update only subplot
            self.fig.canvas.update()
            self.fig.canvas.flush_events()
