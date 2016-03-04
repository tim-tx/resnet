#!/usr/bin/env python

import scipy.sparse
import numpy as np
import graph_tool.all as gt
from numpy.random import binomial
import logging as log
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import argparse

parser = argparse.ArgumentParser(description='resistor network solver')
parser.add_argument('edge', type=int, default=100, help='number of resistors on an edge')
args = parser.parse_args()

N = args.edge

log.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=log.INFO)

def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

def csr_rows_set_nz_to_val(csr, rows, value=0):
    for row in rows:
        csr_row_set_nz_to_val(csr, row)
    if value == 0:
        csr.eliminate_zeros()

class MyVisitor(gt.DFSVisitor):
    def __init__(self,visited):
        self.visited = visited

    def discover_vertex(self,u):
        self.visited.add(int(u))

Lx = N+1
Ly = N+1
Lz = N+1

l = Lx*Ly*Lz

rm = 1.0
gm = 0.5

num_bonds = (Lx-1)*Ly*Lz + Lx*(Ly-1)*Lz + Lx*Ly*(Lz-1)

p = 0.5

log.info("generating graph")
g = gt.lattice([Lx,Ly,Lz])

assert(g.num_edges() == num_bonds)

prop = g.new_edge_property('bool',vals=binomial(1,p,num_bonds).astype('bool'))
g.set_edge_filter(prop)

num_vert = g.num_vertices()
g.add_vertex(2)
g.add_edge_list([[num_vert,n] for n in range(Lx*Ly)])
g.add_edge_list([[num_vert+1,n] for n in range(l-Lx*Ly,l)])

log.info("searching")
vcc_path = set()
gt.dfs_search(g,num_vert,visitor=MyVisitor(vcc_path))

print(num_vert+1 in vcc_path)

vcc_path.remove(num_vert)
vcc_path.remove(num_vert+1)

g.remove_vertex(num_vert+1)
g.remove_vertex(num_vert)

plane = [i for i in range(Lx*Ly,2*Lx*Ly+1) if i in g.vertex(i-Lx*Ly).all_neighbours()]

top = range(Lx*Ly)
bot = range(l-Lx*Ly,l)

others = vcc_path-set(top)-set(bot)
fixed = set(range(l))-others

adj = gt.adjacency(g)
del g

csr_rows_set_nz_to_val(adj,fixed)

diag = adj.sum(axis=1)

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

# i don't know if this does anything
#del g

log.info("assembling")
A.assemblyBegin()
A.assemblyEnd()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
#ksp.setType('preonly')
# pc=ksp.getPC()
# pc.setType('lu')
# pc.setFactorSolverPackage('mumps')

ksp.setFromOptions()
log.info("solving with %s" % ksp.getType())
ksp.solve(b,x)
log.info("converged in %d iterations" % ksp.getIterationNumber())

V = x.array
Itot = np.sum([(1.0-V[i])*gm for i in plane])
log.info("total current %f" % Itot)
emt = (1.-(1.-p)*3./2.)/N*(N+1)**2
log.info("emt prediction %f" % emt)
