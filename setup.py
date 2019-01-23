#!/usr/bin/env python

from calie.__init__ import __version__
from setuptools import setup, find_packages


setup(name='calie',
      version=__version__,
      description='Toolkit for vector fields manipulations.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/SebastianoF/calie',
      packages=find_packages())
