import os
from os.path import join as jph


root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

root_data = '/Users/sebastiano/a_data/TData/'

pfo_adni = jph(root_data, 'ADNI_longitudinal')
pfo_brainweb = jph(root_data, 'BrainWeb')

assert os.path.exists(pfo_adni)
assert os.path.exists(pfo_brainweb)

pfo_output_A1     = jph(root_data, 'A1_bench')
pfo_output_A1_3d  = jph(root_data, 'A1_bench_3d')
pfo_output_A4_SE2 = jph(root_data, 'A4_bench_SE2')
pfo_output_A4_HOM = jph(root_data, 'A4_bench_HOM2')
pfo_output_A4_GL2 = jph(root_data, 'A4_bench_GL2')
pfo_output_A4_GAU = jph(root_data, 'A4_bench_GAU')
pfo_output_A4_BW  = jph(root_data, 'A4_bench_BW')
pfo_output_A4_AD  = jph(root_data, 'A4_bench_AD')
pfo_output_A5_3T  = jph(root_data, 'A5_IC_SA_SE')

assert os.path.exists(pfo_output_A1)
assert os.path.exists(pfo_output_A1_3d)
assert os.path.exists(pfo_output_A4_SE2)
assert os.path.exists(pfo_output_A4_GL2)
assert os.path.exists(pfo_output_A4_GAU)
assert os.path.exists(pfo_output_A4_BW)
assert os.path.exists(pfo_output_A4_AD)
assert os.path.exists(pfo_output_A5_3T)
