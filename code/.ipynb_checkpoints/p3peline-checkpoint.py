import crc
import sys

scenario = str(sys.argv[1])
n        = int(sys.argv[2])
modes    = ['2,2', '2,1', '3,3', '4,4', '4,3']
# modes    = ['4,3', '3,2']
# modes    = ['3,2']

for mode in modes:
    crc.p3peline(scenario, n, mode)


