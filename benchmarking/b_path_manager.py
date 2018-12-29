import os
from os.path import join as jph


root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

root_data = '/Users/sebastiano/a_data/TData/'

pfo_adni = jph(root_data, 'ADNI_longitudinal')
pfo_brainweb = jph(root_data, 'BrainWeb')

pfo_output_A1    = jph(root_data, 'A1_bench')
pfo_output_A1_3d = jph(root_data, 'A1_bench_3d')
pfo_output_A4    = jph(root_data, 'A4_bench')
pfo_output_A5    = jph(root_data, 'A5_bench')
pfo_output_A6    = jph(root_data, 'A6_bench')


assert os.path.exists(pfo_adni)
assert os.path.exists(pfo_brainweb)
assert os.path.exists(pfo_output_A1)
assert os.path.exists(pfo_output_A4)
assert os.path.exists(pfo_output_A5)
assert os.path.exists(pfo_output_A6)
