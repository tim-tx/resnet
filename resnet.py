import scipy.sparse
import numexpr as ne
import numpy as np

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
