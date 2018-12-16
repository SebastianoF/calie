# VECtorsToolkit

Python 3.7 back compatible with 2.7.

Research code for testing and comparing numerical integrations of vector fields in <img src="https://latex.codecogs.com/svg.latex?" title="\mathbb{R}^D"/>.


<!--to obtain the (vectorized) high-resolution 3D MRI volume $`\vec{x}\in\mathbb{R}^N`$ from multiple, possibly motion corrupted, low-resolution stacks of (vectorized) 2D MR slices $`\vec{y}_k \in\mathbb{R}^{N_k}`$ with $`N_k\ll N`$ for $`k=1,...,\,K`$-->


+ [What](https://github.com/SebastianoF/VECtorsToolkit/wiki/What)
+ [Why](https://github.com/SebastianoF/VECtorsToolkit/wiki/Why)
+ [How](https://github.com/SebastianoF/VECtorsToolkit/wiki/How)

## Install

+ Install the code in development mode in a newly created virtualenv.
```
virtualenv -p <local python 3 interpreter> --always-copy <folder with your venvs>
source <folder with your venvs>/bin/activate
git clone <this repository>
cd VECtorsToolkit
pip install -r requirements.txt
pip install -e .
```

+ Install [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) (not fundamental, only to reproduce some part of the benchmarking)


## Testing

Run the tests:
```
pytest
```

Run the test and generate the coverage report:
```
pytest --cov --cov-report html
coverage html
open htmlcov/index.html
```

## Licence 

The code is licenced under [BSD 3-Clause](https://github.com/SebastianoF/VECtorsToolkit/blob/master/LICENCE.txt). 

## Acknowledgements

+ This repository is developed within the GIFT-surg research project.
+ This work was supported by Wellcome / Engineering and Physical Sciences Research Council (EPSRC) 
[WT101957; NS/A000027/1; 203145Z/16/Z]. Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral 
Training in Medical Imaging (EP/L016478/1) and Doctoral Training Grant (EP/M506448/1).