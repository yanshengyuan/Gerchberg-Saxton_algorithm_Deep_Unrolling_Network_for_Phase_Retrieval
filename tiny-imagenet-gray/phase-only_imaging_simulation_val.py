'''
2024.6.26, Shengyuan Yan, TU/e, Eindhoven, NL.
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

input_path='phase_objects_val'

#Grid size, wavelength
#Grid size, wavelength
wavelength=632.8*nm
size=5.12*mm
N=64

#propagate distance
z=400*mm

#Gaussian beam source
F_init=Begin(size,wavelength,N)
I_init=np.full((N, N), 255, dtype=np.float64)
F_init=SubIntensity(F_init, I_init)

objs=os.listdir(input_path)

for runCnt in range(len(objs)):
    obj_path=os.path.join(input_path, objs[runCnt])
    obj_phase=np.load(obj_path)
    #obj_phase=(obj_phase/3)*20
    F=F_init
    F=SubPhase(F, obj_phase)
    
    '''
    plt.imshow(Phase(F), cmap='gray')
    plt.colorbar()
    break
    '''
    
    F=Forvard(F, z, usepyFFTW=True)
    I=Intensity(F, flag=2)
    
    '''
    plt.imshow(I, cmap='gray')
    plt.colorbar()
    break
    '''

    #plt.imshow(Phase(F), cmap='gray')
    #plt.colorbar()
    with lock:
        np.save('I_val/npy/'+objs[runCnt], I)
        plt.imsave('I_val/img/'+objs[runCnt][:-4]+'.png', I, cmap='gray')
        time.sleep(1)
    print(runCnt)
    
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)