###

NAME = 'M1_runAll' 
PROJECT = 'lascon'
PYTHON_VERSION = '3.6'


"""
 ------RUN M1 SIMULATION MODULES----

"""


from netpyne import sim  # import netpyne init module
import os, re
import importlib

## Set working directory  
"""
The code below will traverse the path upwards until it finds the root folder of the project.
"""

workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)
import sys
sys.path.insert(0, './')
import prms_dysp1 as prms  # import parameters file


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
# sim.checkOutput('prms')

#%%

"""
---- VECTOR VALUES SIMULATED DATA-------
 """
 
import numpy as np
import matplotlib.pyplot as plt
datos = np.load('./pipeline/'+NAME+'/store/vdysp.npy')
data_T = np.load(os.path.join(pipeline, 'store', 't_dysp.npy'))



#%%
### GRÁFICOS

from matplotlib import pyplot 
import pandas as pd

# Tendencia de series de tiempo
plt.figure(1)
plt.clf()
plt.plot(datos.T)

### RASTER PLOT
series = [] 
 
for ini in range(0,datos.shape[0]): 
    series.append(datos[ini,:]) 
 
series = np.array(series) 
series = pd.DataFrame(series) 
 
pyplot.matshow(series, interpolation=None, aspect ='auto', cmap='bone', vmin = 1, vmax = 1.05) 
pyplot.colorbar() 

#%%

"""
------Connection Inference by Cross correlation------
"""

from scipy import signal
from scipy import stats 

i = 0
datosNorm = datos
f = len(datos[0,:])
init = f//2


def SurrogateCorrData(datos,N=100): #Número de veces en las que se generará las matrices aleatorizadas
    fftdatos=np.fft.fft(datos,axis=-1)
    ang=np.angle(fftdatos)
    amp=np.abs(fftdatos)
    #Cálculo de la matriz de correlación de los datos aleatorizados
    CorrMat=[]
    for i in range(N):
        angSurr=np.random.uniform(-np.pi,np.pi,size=ang.shape)
        angSurr[:,init:]= - angSurr[:,init:0:-1] #trabajamos sólo en dos dimensiones: tiempo y población
        angSurr[:,init]=0
        
        fftdatosSurr=np.cos(angSurr)*amp + 1j*np.sin(angSurr)*amp
    
        datosSurr=np.real(np.fft.ifft(fftdatosSurr,axis=-1)) #arroja la valores reales de los datos aleatorizados
        spcorr2,pval2=stats.spearmanr(datosSurr,axis=1)
        CorrMat.append(spcorr2)
        
    CorrMat=np.array(CorrMat)
    return CorrMat
  

SCM=SurrogateCorrData(datosNorm)     

#Calculate the standart desviation and mean of SCM=SurrogateCorrData
meanSCM=np.mean(SCM,0)
sdSCM=np.std(SCM,0)

# GRÁFICOS DE LAS MATRICES DE CORRELACIÓn

#   Ploteo de las matrices de correlación considerando la desviación estándar (2) de la distribución de la matriz aleatorizada

spcorr,pval=stats.spearmanr(datosNorm,axis=1) 
#spcorr[pval>=0.0001]=0


#Filtro de la matriz original, que tomará como 0 a los valores abs de la correlación que sean menores a 2SD del promedio de SCM, 
 #          Cambiamos a tres derviaciones estándar
spcorr[np.abs(spcorr)<(meanSCM + 2*sdSCM)]=0

# np.savetxt(filename +"Pilospcorr.csv", spcorr, delimiter=',')


plt.figure(4)
plt.clf()

 
plt.subplot(231)
plt.plot(datosNorm.T)

plt.subplot(232)
plt.imshow(spcorr,interpolation='none',cmap='inferno',vmin=-1,vmax=1)
plt.colorbar()
plt.tick_params(axis = 'both', labelsize= 12)
plt.xlabel("nCells", fontsize = 13)
plt.ylabel("nCells", fontsize = 13)
plt.grid(False)    

plt.subplot(233)
plt.hist(spcorr.ravel(),bins=50)

plt.subplot(234)
plt.plot(SCM)

plt.subplot(235)
plt.imshow(np.std(SCM,0),interpolation='none',cmap='viridis')
#plt.imshow(spcorr2,interpolation='none',cmap='jet')

plt.grid(False)    

plt.subplot(236)
plt.hist(SCM[:,5,8],bins=50)


#%%%

"""
------------ CONNECTIVITY MATRIX------
"""


spcorr2=np.tril(spcorr)
z = np.where((spcorr2>0.1) & (spcorr2<0.9)) # forma de usar doble condición en un where

zz=[]

for i1,i2 in zip(z[0],z[1]):
    zz.append((i1,i2, spcorr2[i1,i2]))
zz=np.array(zz)  

