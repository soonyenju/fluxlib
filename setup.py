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
	version = "0.0.8",
	keywords = ("eddy covariance postprocessing, gapfilling and partitioning", "flux"),
	description = "left blank",
	long_description = "left blank",
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