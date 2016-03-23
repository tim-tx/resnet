#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import logging as log
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate a discrete space and calculate its pore size distribution.")
parser.add_argument('p', type=float, default=0.5, help='the desired volume fraction')
args = parser.parse_args()

log.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=log.INFO)

st = time.time()

Nphys = 25 # number of cubes on an edge in physical space
M = 2 # edge length (w.r.t. Nphys) of cubes to be removed, total vol removed is M**3
p = args.p # volume fraction
tol = 1e-3 # desired tolerance on volume fraction for generation

log.info("Nphys = %d, M = %d, p = %f" % (Nphys,M,p))

def gen_flags(fac):
    '''Generate a 1D array of booleans with random values'''
    gen_p = np.clip(fac*p/M**3,0.,1.) # not sure if clipping here would confuse newton's method
    flags = np.random.binomial(1,gen_p,size=Nphys**3).astype('bool')
    return flags

def flags_fill(flags):
    '''Expand a cube of True around singleton True values in flags array'''
    idx = np.argwhere(flags == True)
    max_idx = Nphys**3 - 1
    last = flags[max_idx]

    # find where there is room for expansion in each direction
    i = np.where(idx % Nphys < Nphys-1)[0]
    j = np.where(idx % (Nphys**2) < (Nphys-1)*Nphys)[0]
    k = np.where(idx % (Nphys**3) < (Nphys-1)*Nphys**2)[0]

    # find room for expansion in multiple directions
    ij = np.intersect1d(i,j)
    ik = np.intersect1d(i,k)
    jk = np.intersect1d(j,k)
    ijk = np.intersect1d(ij,k)

    flags[np.clip(idx[i]+1,0,max_idx)] = True # i+1
    flags[np.clip(idx[j]+Nphys,0,max_idx)] = True # j+1
    flags[np.clip(idx[k]+Nphys**2,0,max_idx)] = True # k+1
    flags[np.clip(idx[ij]+1+Nphys,0,max_idx)] = True # i+1, j+1
    flags[np.clip(idx[ik]+1+Nphys**2,0,max_idx)] = True # i+1, k+1
    flags[np.clip(idx[jk]+Nphys*(Nphys+1),0,max_idx)] = True # j+1, k+1
    flags[np.clip(idx[ijk]+1+Nphys*(Nphys+1),0,max_idx)] = True # i+1, j+1, k+1

    # needed?
    flags[max_idx] = last

    return flags

log.info("generating pore space")

def init_newton():
    '''First steps for a Newton's method iteration'''
    fac1 = 1.0 # initialize scaling factor on p to get the desired volume fraction (overlap of bodies removed requires this)
    fac2 = 1.01

    flags = flags_fill(gen_flags(fac1))
    ratio1 = np.sum(flags)/Nphys**3
    # log.info("volume fraction = %f" % ratio1)
    flags = flags_fill(gen_flags(fac2))
    ratio2 = np.sum(flags)/Nphys**3
    # log.info("volume fraction = %f" % ratio2)

    deriv = ((p-ratio1) - (p-ratio2))/(fac1-fac2)
    fac = fac2
    ratio = ratio2
    return fac2,ratio2,deriv

# generate a matrix with True volume fraction close to desired value
fac,ratio,deriv = init_newton()
count = 1
while(np.abs(p-ratio) > tol):
    if deriv == 0:
        fac,ratio,deriv = init_newton()
    fac_old = fac
    fac = fac - (p-ratio)/deriv
    flags_u = gen_flags(fac)
    flags = flags_fill(flags_u.copy())
    ratio_old = ratio
    ratio = np.sum(flags)/Nphys**3
    deriv = ((p-ratio_old) - (p-ratio))/(fac_old-fac)
    count += 1

log.info("finished after %d iterations" % count)
log.info("volume fraction = %f" % ratio)

flags = flags.reshape(Nphys,Nphys,Nphys)

space_edge = 50
side = space_edge / Nphys
r = side / np.sqrt(np.pi)

# walls
centers = (np.argwhere(flags) + 0.5) / Nphys * space_edge

ps = []
C = []
P = []

log.info("calculating pore size distribution")

def gen_p():
    return np.random.rand(3) * space_edge
def bubble_radius(c,sign=-1.0):
    # if np.isnan(c).any():
    #     return 0.0
    s =  cdist(centers,c[None,...])
    ret = sign*(np.min(s) - r)
    return ret


for i in range(100):
    P.append(gen_p())
    while( (cdist(P[-1][None,...],centers) <= r*1.3).any() ):
        P[-1] = gen_p()

    cons = ({'type': 'ineq',
             'fun' : lambda x: np.array(bubble_radius(x,sign=1.0) - np.sqrt(np.sum((P[-1]-x)**2))) })
    bounds = ((0,space_edge),)*3
    res = minimize(bubble_radius, P[-1], method='SLSQP', constraints=cons, bounds=bounds, options={'disp':False})

    ps.append(res.fun)
    C.append(res.x)


ps = np.array(ps)*-1
P = np.array(P)
C = np.array(C)

samples, step = np.linspace(0,np.max(ps),100,retstep=True)
cuml = np.array([np.where(ps >= x)[0].shape[0] for x in samples])/ps.shape[0]

PSD = -np.gradient(cuml,step)

#plt.plot(samples,cuml)
#plt.bar(samples,PSD,step)
plt.plot(samples,PSD*10)
#plt.hist(ps,20)
plt.xlabel('pore radius')
plt.ylabel('probability density')
plt.show()

lines = '''# vtk DataFile Version 2.0
Unstructured grid legacy vtk file with point scalar data
ASCII

DATASET UNSTRUCTURED_GRID
'''
lines += "POINTS %d double\n" % centers.shape[0]

for cen in centers:
    lines += "%f %f %f\n" % tuple(cen)

lines += 'POINT_DATA %d\n' % centers.shape[0]
lines += '''SCALARS radii double
LOOKUP_TABLE default
'''
for cen in centers:
    lines += "%f\n" % r

f = open('test.vtk','w')
f.writelines(lines)
f.close()
