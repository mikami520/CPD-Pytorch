'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-03-31 01:27:47
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-03-31 01:55:43
FilePath: /CPD-Pytorch/setup.py
Description: 
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
from setuptools import setup
'''
def readme():
    with open('README.rst') as f:
        return f.read()  
'''
setup(name = 'torchcpd',
      version='0.0.1',
      description='Pure Numpy Implementation of the Coherent Point Drift Algorithm',
      #long_description=readme(),
      url='https://github.com/mikami520/CPD-Pytorch',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
      ],
      keywords='pytorch, image processing, point cloud, registration, mesh, surface',
      author='Yuliang Xiao',
      author_email='yl.xiao@mail.utoronto.ca',
      license='Apache',
      packages=['torchcpd'],
      install_requires=['torch', 'future', 'matplotlib'],
      zip_safe=False
      )
