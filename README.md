# VECtorsToolkit

Python 2.7.1.
 

## Introduction

Small framework to manipulate, integrate, derivate and visualize vector fields.
In includes local operations from Lie group theory as Lie logarithm, exponential, log-composition, exp-sum and 
exp-multiplication by a scalar (see docs for definitions).
It is a light refactoring of the less flexible but more specific software proposed in the repository lie_exponential.

The main aim other than provide prototyping playground where to test algorithms for the manipulations of vector fields, 
as well as to compute statistics, and test new methods on computing statistics on transformations.

## What is a vector field?

+ A vector field (vf) is here defined as a 5-dimensional numpy array (x, y, z, t, d) providing a discrete representation 
of a mapping from R^d x [0,T] -> R^d, sampled on a regular grid (omega). The time interval [0,T] is discretised as well 
in t different timepoints. If t=0 then the vector field is stationary (svf) otherwise is time varying (tvvf).  

+ To keep the software as simple as possible, a vf is just a numpy array passing a sanity test that checks its dimension.

+ This tool offers a range of functions to manipulate vector fields.

## What can you do with VECtorsToolkit

+ Quick introduction

+ Examples

+ Software design

## Contributions



## Acknowledgements