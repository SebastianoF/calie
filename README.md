# VECtorsToolkit

Python 3.7 back compatible with 2.7.

Research code for testing and comparing numerical integrations of vector fields in 
![img](http://latex.codecogs.com/svg.latex?\mathbb{R}^D).

![Output sample](https://github.com/SebastianoF/VECtorsToolkit/blob/master/docs/figures/deformations.gif)

Rotation of ![img](http://latex.codecogs.com/svg.latex?\pi/8) around the centre of the field of view of an axial
slice from [BrainWeb](http://brainweb.bic.mni.mcgill.ca/brainweb/) dataset.
The transformation is parametrised with stationary velocity field
in Lagrangian coordinates (in red),
whose integral or flow field between 0 and 1 represents the actual transformation (in blue).

![Output sample](https://github.com/SebastianoF/VECtorsToolkit/blob/master/docs/figures/LieExpLog.png)

The stationary velocity fields are in general elements of the infinite dimensional Lie algebra of diffeomorphisms.
A diffeomorphism ![img](http://latex.codecogs.com/svg.latex?\phi) is an element of the infinite dimensional Lie group
over ![img](http://latex.codecogs.com/svg.latex?\Omega), subset of
![img](http://latex.codecogs.com/svg.latex?\mathbb{R}^D).
In the image above a stationary velocity field is represented with an arrow of the tangent space of the group of
diffeomorphisms. Lie exponential and Lie logarithm map the vector field in the corresponding flow and vice versa.

## Documentation

+ [What](https://github.com/SebastianoF/VECtorsToolkit/wiki/What)
+ [Why](https://github.com/SebastianoF/VECtorsToolkit/wiki/Why)
+ [How](https://github.com/SebastianoF/VECtorsToolkit/wiki/How)

## Installing

+ Install the code in development mode in a newly created virtualenv.
```
virtualenv -p <local python 3 interpreter> --always-copy <folder with your venvs>
source <folder with your venvs>/bin/activate
git clone <this repository>
cd VECtorsToolkit
pip install -r requirements.txt
pip install -e .
```

+ Install [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) (to reproduce advanced benchmarking)


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