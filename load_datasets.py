#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:21:07 2021

@author: mike
"""
import os
import requests
import yaml

####################################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)


remotes = param['remotes']

url = 'http://tethys-ts.xyz/tethys/data/add_datasets'

##################################################
### Load datasets

resp = requests.post(url, json=remotes)

print(resp.content)
