import numpy as np
import crc
import sys
import time

scenario = str(sys.argv[1])
n        = int(sys.argv[2])
# n = 0
# mode     = str(sys.argv[3])
modes  = ['2,2', '2,1', '3,3', '3,2', '4,4', '4,3']
for mode in modes:
    t_i = time.time()
    crc.generate_waveforms_SEOBNRv5_to_save(scenario, n, mode)
    print(f"\n{time.time() - t_i:.3f} seconds\n")
