import numpy as np
import skymodel

ucoeffs = np.loadtxt('coeffs_u.txt')

sm_u=skymodel.SkyModel(ucoeffs)
print (sm_u(sunAz=30, sunAlt=-30, objAz=130, objAlt=40))
print (sm_u(sunAz=30, sunAlt=-20, objAz=130, objAlt=40))
print (sm_u(sunAz=30, sunAlt=-15, objAz=130, objAlt=40))
print (sm_u(sunAz=30, sunAlt=-12, objAz=130, objAlt=40))

