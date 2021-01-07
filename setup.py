import sys, os
from setuptools import setup, find_packages

'''
Optimal Agents Setup Script
Notes for later: see package_data arg if additional files need to be supplied.
'''
if sys.version_info.major != 3:
    print('Please use Python3!')

setup(name='optimal_agents',
        packages=[package for package in find_packages()
                    if package.startswith('optimal_agents')],
        install_requires=[
           ],
        extras_require={
            },
        description='Framework for Optimal Agents Experiments',
        author='Joey Hejna',
        version='0.0.1',
        )
