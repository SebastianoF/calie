# VECtorsToolkit

Python 2.7.1.
Debugging in progress... Do not use this brain!

## Introduction

Small framework to manipulate, integrate, derivate and visualize vector fields.
In includes local operations from Lie group theory as Lie logarithm, exponential, log-composition, exp-sum and 
exp-multiplication by a scalar (see docs - in progress -  for definitions).
It is a light refactoring of the less flexible but more specific software proposed in the repository lie_exponential.

The main aim other than provide prototyping playground where to test algorithms for the manipulations of vector fields, 
as well as to compute statistics, and test new methods on computing statistics on transformations.

## What is a vector field?

+ A vector field (vf) is here defined as a 5-dimensional numpy array (x, y, z, t, d) providing a discrete representation 
of a mapping from R^d x [0,T] -> R^d, sampled on a regular grid (omega). The time interval [0,T] is discretised as well 
in t different timepoints. If t=0 then the vector field is stationary (svf) otherwise is time varying (tvvf).  

+ To keep the software as simple as possible, a vf is just a numpy array passing a sanity test that checks its dimension.

+ This tool offers a range of functions to manipulate vector fields.


## What can a vector field represent?

A vector field can represent: 

 + SVF stationary velocity field, arguments of the Lie exponential map. They live in the Lie agebra and admit sum, scalar product and inner product. Usually indicated with u, v.
 + Flow (of diffeomorphisms) for the position after being in the SVF for a unit of time, i.e. the output of the exponential map. They admit composition, forming a group with it. Usually indicated with phi.
 + Flow (of volumorphism) as the flow, with Jacobian constrained to be equals to 1.

A vector field represent an SVF properly, and a Flow improperly. In this second case in fact provides only the staring and ending point of the flow, regardless its integral curve or path.
Each can be expressed in two coordinate system:

 + Eulerian coordinates (whose identity has the coordinates of the pixel at each pixel)
 + Lagrangian coordinates (whose identity is represented by a matrix everywhere zero)

An SVF can be composed with a flow, and this is the second member of the stationary ODE.
A flow can be composed with an image (or scalar field) and the result is the warped image.

In the code, SVF and Flows are always represented in Lagrangian coordinates, passing to Eulerian system only when need to apply the composition.

The **deformation** (vector field in Eulerian coordinates and Flow) and **displacement** (vector field in Lagrangian coordinates but also svf) nomenclature have been removed entirely to avoid confusion. 
The proposed naming convention is closer to original hydrodynamics papers.


## What can you do with VECtorsToolkit

+ Quick introduction

+ Examples

+ Software design

## Contributions



## Acknowledgements