# SkyNEt imports
from SkyNEt.instruments import InstrumentImporter
import config_switch_to_matrix as config

# Initialize config object
cf = config.experiment_config()

# Initialize serial object
ser = InstrumentImporter.switch_utils.init_serial(cf.comport)

# Switch to matrix
matrix=cf.matrix()
InstrumentImporter.switch_utils.connect_matrix(ser, matrix)

# Status print
print(f'Connected matrix!')