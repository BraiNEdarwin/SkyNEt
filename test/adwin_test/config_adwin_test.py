import numpy as np

class experiment_config(object):
    '''
    This is the configuration for testing the ADwin.
    The experiment is done as follows:
    - connect AO 1-4 directly to AI [2, 4, 6, 8]

    Different sine waves will be presented to each input and plots
    are made afterwards such that you can judge if the data is the 
    same.
    '''

    def __init__(self):
        self.fs = 1000
        self.n_points = 10000
        self.frequency = 1  # Hz

    def Generate_input(self):
        t = np.linspace(0, self.n_points/self.fs, self.n_points)
        x = np.zeros((4, self.n_points))
        for i in range(4):
            x[i] = np.sin(2*np.pi*self.frequency*t + 0.25*np.pi*i)

        return t, x
