import numpy as np
import crc
import sys

N          = int(sys.argv[1])
MODES      = ['2,2', '2,1', '3,3', '3,2', '4,4', '4,3']
COMPONENTS = ['A', 'phi']

mode = sys.argv[2]
for component in COMPONENTS:
    crc.p2peline_spinless(mode, component, N, spin=False, saving=True)