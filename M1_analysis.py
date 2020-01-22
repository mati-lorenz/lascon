#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:33:12 2020

@author: matias
"""

NAME = 'M1_analysis' 
PROJECT = 'lascon'
PYTHON_VERSION = '3.6'

import numpy as np
import json
import os, re


## Set working directory  
"""
The code below will traverse the path upwards until it finds the root folder of the project.
"""

workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)


## Set  up pipeline folder if missing  
"""
The code below will automatically create a pipeline folder for this code file if it does not exist.
"""

pipeline = os.path.join('2_pipeline', NAME)
    
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))


with open('./data/M1.json') as f:
    data = json.load(f)
    
    time = data['simData']['t']
    
    voltajes = np.zeros((284, 100000))
    for key in data['simData']['V']:
        voltajes[int(key.strip('cell_'))] = np.array(data['simData']['V'][key])


    np.save('t_control', time)
    np.save('v_control', voltajes)

    posiciones = np.array([[cell['tags']['xnorm'], cell['tags']['ynorm']] 
                            for cell in data['net']['cells']])