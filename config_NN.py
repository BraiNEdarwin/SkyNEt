# Config file for the experiments

# Benchmarks settings
benchmark = ['bl', 'AND']
  #'wr' for waveform regression benchmark
WavePeriods = 15
WaveFrequency = 8.5
# WavePeriods2 = 0
WaveFrequency2 = 18.5


# I/O settings
SampleFreq = 1000
skipstates = 0

# Evolution settings
genes = 5
genomes = 25
generations = 500
generange = [[-900,900], [-900, 900], [-900, 900], [-900, 900], [-900, 900], [0.1, 2]]
genelabels = ['CV1/T11','CV2/T13','CV3/T17','CV4/T7','CV5/T1', 'Input scaling']
fitnessAvg = 1  #amount of runs per genome
fitnessParameters = [1, 0, 0, 1]

#spiralsettings
SpiralOffset = 0
n_points = 1000

# input1 = t1/e5
# input2 = t3/e6
# output = t11/e1

# Save settings
filepath = 'TEST_Evolution_NN/'
name = benchmark[1]
