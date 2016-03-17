#!/usr/bin/env python

import numpy as np
import numexpr as ne
import graph_tool.all as gt
from scipy.sparse import triu
import time
import logging as log
import argparse
import resnet

parser = argparse.ArgumentParser(description="Generate a discrete, randomly 'porous' space and a matching resistor network. Save the edge filter for graph-tool and generate Java for COMSOL.")
parser.add_argument('p', type=float, default=0.5, help='the desired volume fraction')
parser.add_argument('N', type=int, default=100, help='the number of lattice sites on an edge for the resistor network')
args = parser.parse_args()

log.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=log.INFO)

st = time.time()

Nphys = 25 # number of cubes on an edge in physical space
Nres = args.N # number of lattice sites on an edge
M = 2 # edge length (w.r.t. Nphys) of cubes to be removed, total vol removed is M**3
p = args.p # volume fraction
tol = 1e-3 # desired tolerance on volume fraction for generation

log.info("Nphys = %d, Nres = %d, M = %d, p = %f" % (Nphys,Nres,M,p))

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


def init_newton():
    '''First steps for a Newton's method iteration'''
    fac1 = 1.0 # initialize scaling factor on p to get the desired volume fraction (overlap of bodies removed requires this)
    fac2 = 1.01

    flags = flags_fill(gen_flags(fac1))
    ratio1 = np.sum(flags)/Nphys**3
    log.info("volume fraction = %f" % ratio1)
    flags = flags_fill(gen_flags(fac2))
    ratio2 = np.sum(flags)/Nphys**3
    log.info("volume fraction = %f" % ratio2)

    deriv = ((p-ratio1) - (p-ratio2))/(fac1-fac2)
    fac = fac2
    ratio = ratio2
    return fac2,ratio2,deriv

# generate a matrix with True volume fraction close to desired value
fac,ratio,deriv = init_newton()
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
    log.info("volume fraction = %f" % ratio)

flags = flags.reshape(Nphys,Nphys,Nphys)

log.info("generating lattice")

g = gt.lattice([Nres,Nres,Nres])

mat = triu(gt.adjacency(g))

bonds = np.vstack([mat.row,mat.col])

r1 = 8e-9
r2 = 25e-9

log.info("warping lattice")
x,y,z = resnet.bonds_to_xyz(bonds,Nres,r1,r2)

# coordinates inside a certain radius
# inner = np.argwhere(np.sqrt(x**2 + y**2 + z**2) < (r1 + 0.2*(r2-r1))).flatten()

# fit lattice into space of 'physical matrix' indices
x = ne.evaluate('(x/r2/2+0.5)*(Nphys-1)')
y = ne.evaluate('(y/r2/2+0.5)*(Nphys-1)')
z = ne.evaluate('(Nphys-1)*(z/r2+1)')

# now each bond maps onto a boolean
x = np.round(x).astype(np.int64)
y = np.round(y).astype(np.int64)
z = np.round(z).astype(np.int64)

#prop = g.new_edge_property('bool',vals=flags[x,y,z])
#g.set_edge_filter(prop)
#g.save('graph_'+time.strftime("%y%m%d_%H%M%S")+'.gt',fmt='gt')
resmask = flags[x,y,z]
stamp = time.strftime("%y%m%d_%H%M%S")
np.save('mask_'+stamp,resmask)
np.save('flags_'+stamp,flags)

#vf_inner = np.sum(resmask[inner])/resmask[inner].shape[0]
vf_full = np.sum(resmask)/resmask.shape[0]
log.info("lattice volume fraction = %f" % vf_full)

log.info("total time %f sec" % (time.time()-st))

st = time.time()

r2 *= 1e9
r1 *= 1e9

# build the first set of cubes, the inclusions

sizes = np.array([2*r2/Nphys,r2/Nphys])

loc = np.argwhere(flags == True).astype(np.float)
loc[:,:2] = (loc[:,:2]-0.5*(Nphys-1))/(0.5*Nphys)*r2
loc[:,2] = (loc[:,2]+0.5)/Nphys*r2
# remove the stuff that's outside r1 and r2
loc = loc[ np.sqrt(loc[:,0]**2+loc[:,1]**2+loc[:,2]**2) < r2 ]
loc = loc[ np.sqrt(loc[:,0]**2+loc[:,1]**2+loc[:,2]**2) > r1 ]
loc = loc.astype(np.str)

java = '    model.geom("geom1").feature("blk1").set("pos", new String[]{"0.0", "0.0", "0.0"});\n'
java += '    model.geom("geom1").feature("blk1").set("size", new String[]{"'
java += np.str(sizes[0]) + '", "' + np.str(sizes[0]) + '", "' + np.str(sizes[1])
java += '"});\n    model.geom("geom1").create("copy1", "Copy");\n    model.geom("geom1").feature("copy1").set("displx", "'
java += ','.join(loc[:,0])
java += '");\n    model.geom("geom1").feature("copy1").set("disply", "'
java += ','.join(loc[:,1])
java += '");\n    model.geom("geom1").feature("copy1").set("displz", "'
java += ','.join(loc[:,2])
java += '");\n'

file = open('cube-java-1.txt','w')
file.writelines(java)
file.close()

# build second set, the cobes that fill the remaining space

sizes = np.array([2*r2/Nphys,r2/Nphys])

loc = np.argwhere(flags == False).astype(np.float)
loc[:,:2] = (loc[:,:2]-0.5*(Nphys-1))/(0.5*Nphys)*r2
loc[:,2] = (loc[:,2]+0.5)/Nphys*r2
# remove the stuff that's outside r1 and r2
loc = loc[ np.sqrt(loc[:,0]**2+loc[:,1]**2+loc[:,2]**2) < r2 ]
loc = loc[ np.sqrt(loc[:,0]**2+loc[:,1]**2+loc[:,2]**2) > r1 ]
loc = loc.astype(np.str)

java = '    model.geom("geom1").feature("blk2").set("pos", new String[]{"0.0", "0.0", "0.0"});\n'
java += '    model.geom("geom1").feature("blk2").set("size", new String[]{"'
java += np.str(sizes[0]) + '", "' + np.str(sizes[0]) + '", "' + np.str(sizes[1])
java += '"});\n    model.geom("geom1").create("copy2", "Copy");\n    model.geom("geom1").feature("copy2").set("displx", "'
java += ','.join(loc[:,0])
java += '");\n    model.geom("geom1").feature("copy2").set("disply", "'
java += ','.join(loc[:,1])
java += '");\n    model.geom("geom1").feature("copy2").set("displz", "'
java += ','.join(loc[:,2])
java += '");\n'

file = open('cube-java-2.txt','w')
file.writelines(java)
file.close()

log.info("built java code and wrote it in %f sec" % (time.time()-st))
