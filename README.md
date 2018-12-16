# VECtorsToolkit

Python 3.7 back compatible with 2.7.

## Introduction

Research code for the numerical integration of vector fields.
The main aim of this code is to test and compare a range of algorithms for the integration of vector fields.

## Documentation

+ [What](https://github.com/SebastianoF/VECtorsToolkit/wiki/What)
+ [Why](https://github.com/SebastianoF/VECtorsToolkit/wiki/Why)
+ [How](https://github.com/SebastianoF/VECtorsToolkit/wiki/How)

## Install

Install the code in development mode in a newly created virtualenv.
```
virtualenv -p <local python 3 interpreter> --always-copy <folder with your venvs>
source <folder with your venvs>/bin/activate
git clone <this repository>
cd VECtorsToolkit
pip install -r requirements.txt
pip install -e .
```

Install [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) (to reproduce some part of the benchmarking)


## Testing

Run the tests:
```
pytest
```

Run the test and generate the report:
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