#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Songyan Zhu
# Mail: sz394@exeter.ac.uk
# Created Time:  2020-05-08
#############################################


from setuptools import setup, find_packages

setup(
	name = "fluxlib",
	version = "0.0.22",
	keywords = ("eddy covariance", "flux", "gap-filling"),
	description = "fluxlib is a package for eddy covariance post-processing",
	long_description = "Three parts: 1) fluxnet2015 dataset processing; 2) gap-filling; 3) partitioning refferred to Hesseflux please",
	license = "MIT Licence",

	url = "https://github.com/soonyenju/fluxlib",
	author = "Songyan Zhu",
	author_email = "sz394@exeter.ac.uk",

	packages = find_packages(),
	include_package_data = True,
	platforms = "any",
	install_requires=[

	]
)