#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import logging as log
import argparse
import matplotlib.pyplot as plt
import resnet

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

flags = resnet.discrete_pore_space(Nphys,M,p,tol)

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
