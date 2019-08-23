''''
Measurement script to perform an evolution experiment of a selected
gate. This will initially be tested on the Heliox (with nidaq) setup.
'''

# Import packages



import instruments.InstrumentImporter

import time

instruments.InstrumentImporter.reset(0,0)
