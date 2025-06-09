import crc
import numpy as np
import sys

# Conditions
scenario  = 'NS'
# n         = int(sys.argv[1])
n = 4
mode      = str(sys.argv[2])

# Waveforms
waveforms = crc.generate_waveforms_SEOBNRv5_NS(n, mode)
waveforms = crc.add_A_phi(waveforms)

# RBs & EI
A, phi = crc.reformat(waveforms)[1], crc.reformat(waveforms)[2]
crc.p1peline(A, scenario, n, mode, 'A'  , 1e-10, plotting = True, saving = True)
crc.p1peline(phi, scenario, n, mode, 'phi', 1e-8 , plotting = True, saving = True)

# ANNs
crc.p2peline_A(waveforms, scenario, n, mode, saving = True, plotting = True)
crc.p2peline_phi(waveforms, scenario, n, mode, saving = True, plotting = True)
