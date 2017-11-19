#!/usr/bin/env python

import scipy.sparse
import numpy as np
import graph_tool.all as gt
from numpy.random import binomial
import logging as log
import petsc4py
import sys
#petsc4py.init(sys.argv)
from petsc4py import PETSc
import argparse

import resnet

parser = argparse.ArgumentParser(description='Cubic resistor lattice solver using PETSc. Specify either an npy file for the graph edge filter or generate one randomly for the specified bond probability.')
parser.add_argument('N', type=int, default=100, help='number of lattice sites on an edge, needs to be specified if loading a filter or generating one')
parser.add_argument('--gm', type=float, default=1.0, help='bond conductance')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--maskfile',type=str)
group.add_argument('-p','--probability',type=float,metavar='P',help='probability of placing a bond')
args = parser.parse_args()

log.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=log.INFO)

class MyVisitor(gt.DFSVisitor):
    def __init__(self,visited):
        self.visited = visited

    def discover_vertex(self,u):
        self.visited.add(int(u))

Lx = args.N
Ly = Lz = Lx

l = Lx*Ly*Lz

rm = 1.0
#gm = args.gm
#gm = 1.0 + 1j
gm = (1e-3+1j*2*np.pi*3.1e9*4.4*8.854e-12)*2*np.pi/(1/1e-8-1/25e-9)

num_bonds = (Lx-1)*Ly*Lz + Lx*(Ly-1)*Lz + Lx*Ly*(Lz-1)

log.info("generating graph")
g = gt.lattice([Lx,Ly,Lz])

assert(g.num_edges() == num_bonds)

p = args.probability

if args.maskfile:
    log.info("loading %s" % args.maskfile)
    mask = np.load(args.maskfile)
    p = np.sum(mask)/mask.shape[0]
    vals = mask
else:
    vals = binomial(1,p,num_bonds).astype('bool')

prop = g.new_edge_property('bool',vals=vals)
g.set_edge_filter(prop)

num_vert = g.num_vertices()
g.add_vertex(2)
g.add_edge_list([[num_vert,n] for n in range(Lx*Ly)])
g.add_edge_list([[num_vert+1,n] for n in range(l-Lx*Ly,l)])

log.info("checking for percolation")
vcc_path = set()
gt.dfs_search(g,num_vert,visitor=MyVisitor(vcc_path))

if not num_vert+1 in vcc_path:
    log.warn("not percolating")
else:
    vcc_path.remove(num_vert+1)

vcc_path.remove(num_vert)


g.remove_vertex(num_vert+1)
g.remove_vertex(num_vert)

plane = [i for i in range(Lx*Ly,2*Lx*Ly+1) if i in g.vertex(i-Lx*Ly).all_neighbours() and i in vcc_path]

top = range(Lx*Ly)
bot = range(l-Lx*Ly,l)

others = vcc_path-set(top)-set(bot)
fixed = set(range(l))-others

adj = gt.adjacency(g)
del g

resnet.csr_rows_set_nz_to_val(adj,fixed) # nodes not in vcc_path or those on the top/bottom

diag = np.array(adj.sum(axis=1)).flatten()

log.info("creating PETSc structures")
b = PETSc.Vec().createSeq(l)
x = PETSc.Vec().createSeq(l)
#A = PETSc.Mat().createAIJ(size=adj.shape,nnz=7,csr=(adj.indptr,adj.indices,-adj.data))
A = PETSc.Mat().createAIJ(size=adj.shape,nnz=7)

# the gm factor here and below doesn't actually matter
# scaling each resistor by the same constant still produces
# the same distribution of voltages
A.setValuesCSR(adj.indptr,adj.indices,-gm*adj.data)

for i in fixed:
    A.setValue(i,i,1.0)

for i in others:
    A.setValue(i,i,gm*diag[i])

del adj
    
b.setValuesBlocked(range(Lx*Ly),[1.0]*Lx*Ly)

# i don't know if this actually frees memory
#del g

log.info("assembling")
A.assemblyBegin()
A.assemblyEnd()

ksp = PETSc.KSP().create()
ksp.setOperators(A)

# uncomment if you want direct LU instead of iterative
# ksp.setType('preonly')
# pc=ksp.getPC()
# pc.setType('lu')
# pc.setFactorSolverPackage('mumps')

ksp.setFromOptions()
log.info("solving with %s" % ksp.getType())
ksp.solve(b,x)
log.info("converged in %d iterations" % ksp.getIterationNumber())

V = x.array
Itot = np.sum([(1.0-V[i])*gm for i in plane])
#log.info("total current %e" % Itot)
log.info('total current ({0:.4e} {1} {2:.4e}i)'.format(Itot.real, '+-'[Itot.imag < 0], abs(Itot.imag)))
#emt = gm*(1.-(1.-p)*3./2.)/(args.N-1)*(args.N)**2
emt = 0.125*(-2*gm+6*gm*p+np.sqrt(gm**2*(-2+6*p)**2))/(args.N-1)*args.N**2
log.info('emt prediction ({0:.4e} {1} {2:.4e}i)'.format(emt.real, '+-'[emt.imag < 0], abs(emt.imag)))
#log.info("emt prediction %e" % emt)
