#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: const.py
@time: 2024-01-04
@contact: danerlt001@gmail.com
@desc: 
"""
from pathlib import Path

current_path = Path(__file__)

common_path = current_path.parent

SRC_PATH = common_path.parent

ROOT_PATH = SRC_PATH.parent

DATA_PATH = ROOT_PATH.joinpath("data")
