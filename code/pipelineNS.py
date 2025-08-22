"""
File: pipelineNS.py
Author: Connor Campbell
Description:
    Performs the whole offline stage for the non-spinning ('NS') scenario, for 
    a given mode (mode), for a given parameter space (defined in 
    make_params.ipynb, corresponding with dataset n).

Workflow:
    1. Define some key parameters:
       - scenario: non-spinning ('NS')
       - n: dataset (was working with multiple datasets of different sizes for 
       testing purposes)
       - mode: l,m weighted spherical harmonics, from {'2,1', '2,2', '3,2', 
       '3,3', '4,3', '4,4'}

    2. generate_waveforms_SEOBNRv5_NS() generates a dictionary of full 
    waveforms using the fiducial model (SEOBNRv5HM) from the given set of 
    parameters, for the non-spinning scenario, for the given mode.

    3. add_A_phi() separates the amplitude (A) and phase (phi) components of 
    the full waveforms, and adds them to the dictionary of waveforms.

    4. reformat() separates the amplitude and phase components into
    separate objects, ready for independent processing.

    5. p1peline() creates the reduced basis and B matrix, yielding the B matrix
    for the amplitude and phase components.

    6. p2peline() creates and trains artificial neural networks to 
    independently estimate the values of the amplitude and phase at the 
    empirical time nodes.
"""

import crc
import sys

# Conditions
scenario = 'NS'             # non-spinning
n        = int(sys.argv[1]) # dataset number
mode     = str(sys.argv[2]) # in {'2,1', '2,2', '3,2', '3,3', '4,3', '4,4'}

# Waveforms
waveforms = crc.generate_waveforms_SEOBNRv5_NS(n=n, mode=mode)
waveforms = crc.add_A_phi(waveforms=waveforms)

# RBs & EI
A, phi = crc.reformat(waveforms)[1], crc.reformat(waveforms)[2]
crc.p1peline(
    waveforms=A, 
    scenario=scenario, 
    n=n, 
    mode=mode, 
    component='A', 
    tolerance=1e-10, 
    plotting = True, 
    saving = True
)
crc.p1peline(
    waveforms=phi, 
    scenario=scenario, 
    n=n, 
    mode=mode, 
    component='phi', 
    tolerance=1e-8, 
    plotting = True, 
    saving = True
)

# ANNs
crc.p2peline_A(
    waveforms=waveforms, 
    scenario=scenario, 
    n=n, 
    mode=mode, 
    saving = True, 
    plotting = True
)
crc.p2peline_phi(
    waveforms=waveforms, 
    scenario=scenario, 
    n=n, 
    mode=mode, 
    saving = True, 
    plotting = True
)
