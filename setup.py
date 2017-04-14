#!/usr/bin/env python
#-*-coding: utf-8 -*-

# setup.py
from setuptools import setup, find_packages

setup(name='movies',
      version='0.1',
      packages=find_packages(),
      description='analysis of movie posters using Keras for fun',
      author='Paul Lagree',
      author_email='p.l@example.com',
      license='MIT',
      install_requires=[
            'keras',
            'h5py'
            ],
      zip_safe=False)
