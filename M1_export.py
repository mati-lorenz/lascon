NAME = 'M1_export' 
PROJECT = 'lascon'
PYTHON_VERSION = '3.6'

import prms_control  # import parameters file
from netpyne import sim  # import netpyne sim module
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


np = prms_control.netParams
print("********************\n*\n*  Note: setting noise to 1, since noise can only be 0 or 1 in NeuroML export currently!\n*\n********************")
np.stimSourceParams['background_E']['noise'] = 1
np.stimSourceParams['background_I']['noise'] = 1

sim.createExportNeuroML2(netParams = np, 
                       simConfig = prms_control.simConfig,
                       reference = 'M1',
                       connections=True,
                       stimulations=True)  # create and export network to NeuroML 2
