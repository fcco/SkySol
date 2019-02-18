#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup, find_packages

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(name='skysol',
      version='0.5',
      description='Sky Imager Surface Solar Irradiance Analysis and Forecast Tool',
      long_description=readme + '\n\n' + history,
      url='git@gitlab.uni-oldenburg.de:babo4723/skyimager.git',
      author='Thomas Schmidt',
      author_email='t.schmidt@uni-oldenburg.de',
      packages=['skysol','skysol.lib','skysol.misc','skysol.visualization','skysol.validation'],
      install_requires=[],
      license="BSD",
      zip_safe=False,
      keywords=['skysol','skyimager','all-sky','skycam','cloudcam'],
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    test_suite='tests',
)
