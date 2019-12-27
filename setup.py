#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = []

test_requirements = []

setup(
    author="Egor Panfilov",
    author_email='egor.v.panfilov@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Framework for knee cartilage and menisci segmentation from MRI",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    name='rocaseg',
    packages=find_packages(include=['rocaseg']),
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    url='https://github.com/MIPT-Oulu/RobustCartilageSegmentation',
    version='0.1.0',
    zip_safe=False,
)
