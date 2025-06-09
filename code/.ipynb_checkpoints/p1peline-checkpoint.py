import numpy as np
import crc
import sys

scenario = str(sys.argv[1])
mode     = sys.argv[2]
n        = int(sys.argv[3])

waveforms = crc.add_A_phi(crc.load_waveforms(scenario, mode, n))
crc.p1peline_spinless(scenario, waveforms, mode, tolerance = 1e-8, n)



