import numpy as np
import skymodel

ucoeffs = np.loadtxt('coeffs_u.txt')

sm_u=skymodel.SkyModel(ucoeffs)

# print the sky level in the u-band in nanomaggies/sq arcsec
print (sm_u(sunAz=30, sunAlt=-30, objAz=130, objAlt=40))
print (sm_u(sunAz=30, sunAlt=-20, objAz=130, objAlt=40))
print (sm_u(sunAz=30, sunAlt=-15, objAz=130, objAlt=40))
print (sm_u(sunAz=30, sunAlt=-12, objAz=130, objAlt=40))

