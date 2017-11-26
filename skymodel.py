import numpy as np

def betw(x, x1, x2): return (x > x1) & (x <= x2)
def cosd(x): return np.cos(np.deg2rad(x))
def sind(x): return np.sin(np.deg2rad(x))



def getXY(objAz= None, objAlt = None, sunAz = None, sunAlt=None):
    # projection of the object from the sky into x,y
    # -1<x<1 0<y<1
    daz = np.asarray(np.abs(objAz - sunAz))
    daz[daz>180] = 360-daz[daz>180]
    # delta azimuth between object and sun (ensure it is between 0 and 160)

    x = cosd(daz) * cosd(objAlt)
    y = sind(daz) * cosd(objAlt)
    return x,y
    

class SkyModel:
    """ Simple class to evaluate the sky brightness
    Usage: 
    > sm = SkyModel([0]*18)
    > sky = sm(objAz=13,sunAz=44,objAlt=33,sunAlt=-40)
    It returns the sky level given the location of the object and Sun
    """
    def __init__ (self, coeffs, polyTimeDeg, polyXYDeg):
        """ Coeffs - array of poly coefficients """
        self.coeffs = coeffs
        self.polyTimeDeg = polyTimeDeg
        self.polyXYDeg = polyXYDeg

    def __call__ (self, objAz=None, objAlt=None, sunAz=None, sunAlt=None):
        x, y = getXY(objAz=objAz, objAlt=objAlt, sunAz=sunAz, sunAlt=sunAlt)
        retsky = 10**func(self.coeffs, x, y, sunAlt, self.polyTimeDeg, self.polyXYDeg)
        return retsky


def func(p, x, y, alt, polyTimeDeg, polyXYDeg):
    # The model for sky is
    # log10(sky) = A + B* (SunAlt-18) + C * (SunAlt-18)^2
    # where A, B, C are themselves spatial polynomials of x,y
    # x,y = cos(deltaAz)*cos(objAlt), sin(deltAz)*cos(objAlt)

    polys=[1+x*0]
    X, Y = x, np.sqrt(x**2+y**2)
    
    for i in range(1,polyXYDeg+1):
        for j in range(i+1):
            polys.append(X**j*Y**(i-j))
    # we construct polylomials of X**i*Y**j 
    # where i+j<=polyXYDeg
    npolys = len(polys)
    minalt = -18
    assert (len(p) == (npolys * (polyTimeDeg + 1)))
    # nail all the sun's altitiudes below minalt to minalt
    alt1 = np.maximum(alt, minalt) - minalt
    res = 0
    for i in range(polyTimeDeg + 1):
        res = res + np.dot(p[i * npolys:(i + 1) * npolys], polys) * alt1**i
    return res
