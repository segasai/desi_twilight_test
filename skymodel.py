import numpy as np

def betw(x, x1, x2): return (x > x1) & (x <= x2)
def cosd(x): return np.cos(np.deg2rad(x))
def sind(x): return np.sin(np.deg2rad(x))



class SkyModel:
    """ Simple class to evaluate the sky brightness
    Construction 
    sm = SkyModel([0]*18)
    sky = sm(objAz=13,sunAz=44,objAlt=33,sunAlt=-40)
    """
    def __init__ (self, coeffs=None):
        """ Coeffs - array of poly coefficients """
        self.coeffs = coeffs

    def __call__ (self, objAz=None, objAlt=None, sunAz=None, sunAlt=None):
        daz = np.abs(objAz - sunAz)
        daz[daz>180] = 360-daz
        x = cosd(daz) * cosd(objAlt)
        y = sind(daz) * cosd(objAlt)
        retsky = 10**func(self.coeffs, x, y, sunAlt)
        return retsky

def func(p, x, y, alt):
    # The model for sky is
    # log10(sky) = A + B* (SunAlt-18) + C * (SunAlt-18)^2
    # where A, B, C are themselves spatial polynomials of x,y
    # x,y = cos(deltaAz)*cos(objAlt), sin(deltAz)*cos(objAlt)

    polys = [1 + x * 0, x, y, x**2, y**2, x * y]
    ndeg = 2  # what's the degree of poly w.r.t SunAlt
    npolys = len(polys)
    minalt = -18
    assert (len(p) == (npolys * (ndeg + 1)))
    alt1 = np.maximum(alt, minalt) - minalt
    res = 0
    for i in range(ndeg + 1):
        res = res + np.dot(p[i * npolys:(i + 1) * npolys], polys) * alt1**i
    return res
