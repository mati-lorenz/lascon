
NAME = 'M1_run' 
PROJECT = 'lascon'
PYTHON_VERSION = '3.6'

import prms_dysp1 as prms  # import parameters file
from netpyne import sim  # import netpyne init module
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

pipeline = os.path.join('pipeline', NAME)
    
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store']:
        os.makedirs(os.path.join(pipeline, folder))

sim.createSimulateAnalyze(netParams = prms.netParams, simConfig = prms.simConfig)  # create and simulate network

# check model output
#sim.checkOutput('M1')
