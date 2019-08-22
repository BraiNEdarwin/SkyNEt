from create_binary import bintarget


class VCDimensionConfigs():

    def __init__(self,
                    inputs = [[-1., 0.4, -1., 0.4], [-1., -1., 0.4, 0.4]],
                    dirname = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt',
                    plot = 'True', save = 'True'):
        self.inputs = [[-1., 0.4, -1., 0.4], [-1., -1., 0.4, 0.4]]
        # [[-0.7,0.7,-0.7,0.7,-1.,1.],[-0.7,-0.7,0.7,0.7,0.,0.]]
        self.N = len(self.inputs[0])
        # Create save directory
        # @todo improve the way in which directories are handled
        self.dirname = dirname
        self.plot = plot
        self.save = save
        # Create binary labels for N samples
        # bad_gates = # for N=6 on model [51]
        # ###### On Device ########
        # [55]#[22,23,48,52,53,55,57,60,61] for N=6 w. large range
        # for N=6 with (+/-0.35, 0.) as inputs 5 & 6 w. range +/-[1.2,1.0]: [6,33,37,41,45,53,57,60,61]
        # --> bad gates for N=6 w. range +/-0.9 and lower: [1,3,6,7,9,12,14,17,19,22,23,24,25,28,30,33,35,36,37,38,39,41,44,45,46,47,49,51,52,53,54,55,56,57,60,61,62]
        # binary_labels = bintarget(N)[bad_gates].tolist()
        self.binary_labels = bintarget(self.N).tolist()

        self.threshold = (1-0.5/self.N)  # 1-(0.65/N)*(1+1.0/N)
        # print('Threshold for acceptance is set at: ', self.threshold)
        # Initialize container variables
        self.fitness_classifier = []
        self.genes_classifier = []
        self.output_classifier = []
        self.accuracy_classifier = []
        self.found_classifier = []
