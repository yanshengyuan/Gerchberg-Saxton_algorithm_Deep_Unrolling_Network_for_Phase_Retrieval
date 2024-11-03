'''
2024.5.25, Shengyuan Yan, TU/e, Eindhoven, NL.
'''

import random
from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy.stats import truncnorm
from filelock import FileLock
lock = FileLock('npylock.lock')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

start_time = time.time()

output_path='Simu_Output/'

#Grid size, wavelength
#Grid size, wavelength
wavelength=1000.0*nm
size=5*mm
N=64

#propagate distance
z=0.6*m

#Zernike polynomial aberration
zernikeMaxOrder = 4
zernikeAmplitude = 1.5
zernikeRadius = size/3.5
nollMin = 4
np.random.seed(42)

nollMax = np.sum(range(1,zernikeMaxOrder + 1))  # Maximum Noll index
nollRange=range(nollMin,nollMax+1)
zernikeCoeff=np.zeros(nollMax)

#Gaussian beam source
F_init=Begin(size,wavelength,N)
F_init=GaussBeam(F_init, size/3.5)
I_init=Intensity(F_init)
Phi_init=Phase(F_init)
#plt.imsave(output_path+'init_phase.png' , Phi_init, cmap='rainbow')
#plt.imsave(output_path+'init_intensity.png' , I_init, cmap='jet')
#F_noiseFree=Fresnel(F_init, z, usepyFFTW=True)
#plt.imsave(output_path+'NoiseFree_farfield_intensity.png' , Intensity(F_noiseFree), cmap='jet')

for runCnt in range(10000):
    F=F_init
    zernikeField = F_init
    
    for countNoll in nollRange:        
        (nz,mz) = noll_to_zern(countNoll)
        zernikeCoeff[countNoll-1] = random.uniform(-1,1)*zernikeAmplitude/(nz+1)
        zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
        
        F=zernikeField
    
    Phi_init=Phase(F)
    with lock:
        np.save('train/Phi/'+'Phi-'+str(runCnt)+'.npy', Phi_init)
        time.sleep(1)
    plt.imsave(output_path+'init_phase-' + str(runCnt) + '.png' , Phi_init, cmap='gray')
    F=Fresnel(F, z, usepyFFTW=True)
    I=Intensity(F, flag=2)
    Phi=Phase(F)
    plt.imsave(output_path+'dist_phase' +str(runCnt)+ '.png' , Phi, cmap='gray')
    plt.imsave(output_path+'dist_intensity' +str(runCnt)+ '.png' , I, cmap='gray')
    with lock:
        np.save('train/I/'+'I-'+str(runCnt)+'.npy', I)
        time.sleep(1)
    print(runCnt)
    
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)