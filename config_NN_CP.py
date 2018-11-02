# Config file for the experiments

# Benchmarks settings
benchmark = ['bl', 'XNOR']
  #'wr' for waveform regression benchmark
WavePeriods = 15
WaveFrequency = 8.5
# WavePeriods2 = 0
WaveFrequency2 = 18.5


# I/O settings
SampleFreq = 1000
skipstates = 0

# Evolution settings
genes = 5 # 3 for 4InpV
genomes = 25
generations = 600
generange = [[-900,900], [-900,900], [-900,900], [-900,900], [-900,900]]#, [0.9, 0.9]]
genelabels = ['CV1/T11','CV2/T13','CV3/T17','CV4/T7','CV5/T1', 'Input scaling']
Ftype = 'SF' #'negMSE'# SF is the standard fitness function; otherwise is neg. MSE
fitnessAvg = 1  #amount of runs per genome
fitnessParameters = [1, 0, 0, 0.005]

#spiralsettings
SpiralOffset = 0
n_points = 1000

# input1 = t1/e5
# input2 = t3/e6
# output = t11/e1

# Save settings
filepath = r'../../Evolution_NN_CP_2InpV/'
name = 'CP_inputs_and_targets'
