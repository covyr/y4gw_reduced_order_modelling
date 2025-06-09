import crc
import numpy as np
import sys

# Conditions
scenario  = 'AS'
n         = int(sys.argv[1])
mode      = str(sys.argv[2])
tolA      = float(sys.argv[3])
tolphi    = float(sys.argv[4])

# Waveforms
waveforms = crc.generate_waveforms_SEOBNRv5_AS(n, mode)
waveforms = crc.add_A_phi(waveforms)

# RBs & EI
A, phi = crc.reformat(waveforms)[1], crc.reformat(waveforms)[2]
crc.p1peline(A, scenario, n, mode, 'A'  , tolA , plotting = True, saving = True)
crc.p1peline(phi, scenario, n, mode, 'phi', tolphi , plotting = True, saving = True)

# ANNs
crc.p2peline_A(waveforms, scenario, n, mode, saving = True, plotting = True)
crc.p2peline_phi(waveforms, scenario, n, mode, saving = True, plotting = True)
