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
