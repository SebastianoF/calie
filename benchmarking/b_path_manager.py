import os

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

pfo_adni = ''
pfo_brainweb = ''

pfo_output = '/Users/sebastiano/a_data/TData/A4_bench'


assert os.path.exists(pfo_adni)
assert os.path.exists(pfo_brainweb)
assert os.path.exists(pfo_output)

