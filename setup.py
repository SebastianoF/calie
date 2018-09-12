#!/usr/bin/env python

from VECtorsToolkit.__init__ import __version__
from setuptools import setup, find_packages


setup(name='VECtorsToolkit',
      version=__version__,
      description='Toolkit for vector fields manipulations.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/SebastianoF/VECtorsToolkit',
      packages=find_packages(),
     )
