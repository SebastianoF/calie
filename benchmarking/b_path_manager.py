import os

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

data_adni_longitudinal = ''
data_adni_brainweb = ''

data_output_dir = '/Users/sebastiano/a_data/TData/A4_bench'

assert os.path.exists(data_output_dir)
