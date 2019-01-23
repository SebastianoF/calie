[![coverage](https://github.com/SebastianoF/calie/blob/master/coverage.svg)](https://github.com/SebastianoF/calie/blob/master/coverage.svg)

<p align="center">
<img src="https://github.com/SebastianoF/calie/blob/master/logo_low.png" width="300">
</p>


# calie - crazy about Lie

Python 3.7 back compatible with 2.7.

Research code for testing and comparing numerical integrations of vector fields in 
![img](http://latex.codecogs.com/svg.latex?\mathbb{R}^D) in the infinite dimensional Lie group / Lie
algebra of diffeomorphisms setting.

![Output sample](https://github.com/SebastianoF/calie/blob/master/docs/figures/deformations.gif)

An axial slice from the [BrainWeb](http://brainweb.bic.mni.mcgill.ca/brainweb/) dataset - Subject BW 38
is transformed according to:

+ Translation
+ Rotation of ![img](http://latex.codecogs.com/svg.latex?\pi/8)
+ Unstable node (linear transformation with real positive eigenvalues)
+ Inward spiral (linear transformation, complex conjugate eigenvectors with negative real part)
+ Random homographic transformation

All transformations are parametrised with stationary velocity field
in Lagrangian coordinates (in red),
whose integral or flow field between 0 and 1 represents the actual transformation (in blue).

![Output sample](https://github.com/SebastianoF/calie/blob/master/docs/figures/LieExpLog.png)

A stationary velocity field is an elements of the infinite dimensional Lie algebra of diffeomorphisms
that parametrise the transformation in an Euclidean space.
A diffeomorphism ![img](http://latex.codecogs.com/svg.latex?\phi) is an element of the infinite dimensional Lie group
over ![img](http://latex.codecogs.com/svg.latex?\Omega), subset of
![img](http://latex.codecogs.com/svg.latex?\mathbb{R}^D).
In the figure above, a stationary velocity field is represented with an arrow of the tangent
space of the group of diffeomorphisms.
Lie exponential and Lie logarithm map the vector field in the corresponding flow and vice versa.

## Documentation

+ [What](https://github.com/SebastianoF/calie/wiki/What)
+ [Why](https://github.com/SebastianoF/calie/wiki/Why)
+ [How](https://github.com/SebastianoF/calie/wiki/How)

## Set up

+ [Install](https://github.com/SebastianoF/calie/wiki/How-to-install)
+ [Run unittest](https://github.com/SebastianoF/calie/wiki/Testing)


## Licence 

The code is licenced under [BSD 3-Clause](https://github.com/SebastianoF/calie/blob/master/LICENCE.txt).

## Acknowledgements

+ This repository is developed within the GIFT-surg research project.
+ This work was supported by Wellcome / Engineering and Physical Sciences Research Council (EPSRC) 
[WT101957; NS/A000027/1; 203145Z/16/Z]. Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral 
Training in Medical Imaging (EP/L016478/1) and Doctoral Training Grant (EP/M506448/1).