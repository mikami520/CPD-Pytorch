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
      author_email='yxiao39@jhu.edu',
      license='Apache',
      packages=['torchcpd'],
      #install_requires=['pytorch', 'future'],
      zip_safe=False
      )