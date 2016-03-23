import scipy.sparse
import numexpr as ne
import numpy as np
import logging as log

def _gen_flags(fac,N,M,p):
    '''Generate a 1D array of booleans with random values'''
    gen_p = np.clip(fac*p/M**3,0.,1.) # not sure if clipping here would confuse newton's method
    flags = np.random.binomial(1,gen_p,size=N**3).astype('bool')
    return flags

def _flags_fill(flags,N,M):
    '''Expand a cube of True around singleton True values in flags array'''
    idx = np.argwhere(flags == True)
    max_idx = N**3 - 1
    last = flags[max_idx]

    # find where there is room for expansion in each direction
    i = np.where(idx % N < N-1)[0]
    j = np.where(idx % (N**2) < (N-1)*N)[0]
    k = np.where(idx % (N**3) < (N-1)*N**2)[0]

    # find room for expansion in multiple directions
    ij = np.intersect1d(i,j)
    ik = np.intersect1d(i,k)
    jk = np.intersect1d(j,k)
    ijk = np.intersect1d(ij,k)

    # this is hardcoded for M=2 right now
    flags[np.clip(idx[i]+1,0,max_idx)] = True # i+1
    flags[np.clip(idx[j]+N,0,max_idx)] = True # j+1
    flags[np.clip(idx[k]+N**2,0,max_idx)] = True # k+1
    flags[np.clip(idx[ij]+1+N,0,max_idx)] = True # i+1, j+1
    flags[np.clip(idx[ik]+1+N**2,0,max_idx)] = True # i+1, k+1
    flags[np.clip(idx[jk]+N*(N+1),0,max_idx)] = True # j+1, k+1
    flags[np.clip(idx[ijk]+1+N*(N+1),0,max_idx)] = True # i+1, j+1, k+1

    # needed?
    flags[max_idx] = last

    return flags

def _init_newton(N,M,p):
    '''First steps for a Newton's method iteration'''
    fac1 = 1.0 # initialize scaling factor on p to get the desired volume fraction (overlap of bodies removed requires this)
    fac2 = 1.01

    flags = _flags_fill(_gen_flags(fac1,N,M,p),N,M)
    ratio1 = np.sum(flags)/N**3
    flags = _flags_fill(_gen_flags(fac2,N,M,p),N,M)
    ratio2 = np.sum(flags)/N**3

    deriv = ((p-ratio1) - (p-ratio2))/(fac1-fac2)
    fac = fac2
    ratio = ratio2
    return fac2,ratio2,deriv

def discrete_pore_space(N,M,p,tol):
    """Generate a 3D grid, N x N x N points, remove cubes of size M x M x M. Newton's method is used to get the volume fraction of the space equal to p within tolerance tol."""

    fac,ratio,deriv = _init_newton(N,M,p)
    count = 1
    while(np.abs(p-ratio) > tol):
        if deriv == 0:
            fac,ratio,deriv = _init_newton(N,M,p)
            count += 1
        fac_old = fac
        fac = fac - (p-ratio)/deriv
        flags_u = _gen_flags(fac,N,M,p)
        flags = _flags_fill(flags_u.copy(),N,M)
        ratio_old = ratio
        ratio = np.sum(flags)/N**3
        deriv = ((p-ratio_old) - (p-ratio))/(fac_old-fac)
        count += 1

    log.info("generated space in %d iterations" % count)
    log.info("volume fraction = %f" % ratio)
    flags = flags.reshape(N,N,N)
    return flags


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

def bond_verts_to_ijk_pairs(bonds,Nres):
    """Generate i,j,k coordinate pairs for source and destination vertices for a list of bonds. The list of bonds is assumed to be an (Nres x 2) array specifying source and destination vertices, where vertices have a single coordinate. The output is 3 (Nres x 2) arrays i,j,k specifying the coordinates of source and destination vertices."""
    i = ne.evaluate('bonds % Nres')
    j = ne.evaluate('(bonds%(Nres**2) - i)/Nres').astype(type(i[0,0]))
    k = ne.evaluate('(bonds-bonds%(Nres**2))/(Nres**2)').astype(type(i[0,0]))
    return i,j,k

def bonds_to_xyz(bonds,Nres,r1,r2):
    """Do the coordinate transformation to fit a cubic lattice into a hemisphere with inner radius r1 and outer radius r2. The 'a' parameter relates  Output is x,y,z"""
    i,j,k = bond_verts_to_ijk_pairs(bonds,Nres)

    a = 4 # i don't remember what this does. in my notes somewhere, sorry
    d = a*r1
    l = (r2**3-(d/a)**3)/d**2/3*2*np.pi
    
    x = ne.evaluate('sum(0.5*i,axis=0)')
    y = ne.evaluate('sum(0.5*j,axis=0)')
    z = ne.evaluate('sum(0.5*k,axis=0)')

    x = ne.evaluate('d*(x/(Nres-1) - 0.5)')
    y = ne.evaluate('d*(y/(Nres-1) - 0.5)')
    z = ne.evaluate('l*-z/(Nres-1)')

    r = np.zeros_like(y)
    theta = np.empty_like(y)
    theta.fill(np.pi)

    pi = np.pi

    mask = ne.evaluate('(arctan2(y,x) >= pi/4) & (arctan2(y,x) < 3*pi/4)')
    xm = x[mask]
    ym = y[mask]
    r[mask] = ne.evaluate('2*ym/sqrt(pi)')
    mask = ne.evaluate('mask & (y != 0)')
    theta[mask] = ne.evaluate('pi/2*(1-xm/ym/2)')

    mask = ne.evaluate('(arctan2(y,x) >= 3*pi/4) | (arctan2(y,x) < -3*pi/4)')
    xm = x[mask]
    ym = y[mask]
    r[mask] = ne.evaluate('2*-xm/sqrt(pi)')
    mask = ne.evaluate('mask & (x != 0)')
    theta[mask] = ne.evaluate('pi*(1+ym/xm/4)')

    mask = ne.evaluate('(arctan2(y,x) >= -3*pi/4) & (arctan2(y,x) < -pi/4)')
    xm = x[mask]
    ym = y[mask]
    r[mask] = ne.evaluate('2*-ym/sqrt(pi)')
    mask = ne.evaluate('mask & (y != 0)')
    theta[mask] = ne.evaluate('pi/2*(3-xm/ym/2)')

    mask = ne.evaluate('(arctan2(y,x) >= -pi/4) & (arctan2(y,x) < pi/4)')
    xm = x[mask]
    ym = y[mask]
    r[mask] = ne.evaluate('2*xm/sqrt(pi)')
    mask = ne.evaluate('mask & (x != 0)')
    xm = x[mask]
    ym = y[mask]
    theta[mask] = ne.evaluate('pi*(2+ym/xm/4)')

    phi = theta.copy()

    rho = ne.evaluate('((d/a)**3-z*d*d*3/2/pi)**(1/3)')
    theta = ne.evaluate('pi*(1-r*sqrt(pi)/d/2)')

    x = ne.evaluate('rho*sin(theta)*cos(phi)')
    y = ne.evaluate('rho*sin(theta)*sin(phi)')
    z = ne.evaluate('rho*cos(theta)')

    return x,y,z
