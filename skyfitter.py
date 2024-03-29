import ephem
import astropy.time
import multiprocessing as mp
import numpy as np
import skymodel
import scipy.optimize
import astropy.table as atpy
import scipy.stats

def betw(x, x1, x2): return (x > x1) & (x <= x2)


def cosd(x): return np.cos(np.deg2rad(x))


def sind(x): return np.sin(np.deg2rad(x))


sun = ephem.Sun()
moon = ephem.Moon()
obs = ephem.Observer()
body = ephem.FixedBody()
body._epoch = 2000

# SDSS coordinates

obs.lon = '-105:49:13'  # E+
obs.lat = '32:46:49'  # N+

def getit(ra, dec, x):
    # Compute some extra stuff (asimuths, zd of moon, sun and targets)
    # for a given date x
    ret = {}
    obs.date = x
    body._ra = np.deg2rad(ra)
    body._dec = np.deg2rad(dec)
    moon.compute(obs)
    sun.compute(obs)
    body.compute(obs)

    ret['moonAlt'] = (np.rad2deg(moon.alt))
    ret['moonAz'] = (np.rad2deg(moon.az))
    ret['moonP'] = (moon.phase)

    ret['sunAlt'] = (np.rad2deg(sun.alt))
    ret['sunAz'] = (np.rad2deg(sun.az))

    ret['objAlt'] = (np.rad2deg(body.alt))
    ret['objAz'] = (np.rad2deg(body.az))
    return ret

def getdata(cached=True):
    if not cached:
        import sqlutil
        Q = 'select fieldid, sky_u,sky_g,sky_r,sky_i,sky_z,ramin as ra,decmin as dec,mjd_r as mjd from sdssdr9.field '
        D = sqlutil.get(
            Q, host='cappc127', asDict=True)
        fieldid,sky_u, sky_g, sky_r, sky_i, sky_z, ra, dec, mjd = D.values()

        tab = atpy.Table(data=D)
        tab.write('sdss_sky.fits', format='fits', overwrite=True)
    else:
        tab = atpy.Table().read('sdss_sky.fits')
        fieldid, sky_u, sky_g, sky_r, sky_i, sky_z, ra, dec, mjd = [tab[_] for _
                                                           in 'fieldid,sky_u,sky_g,sky_r,sky_i,sky_z,ra,dec,mjd'.split(',')]
    return fieldid, sky_u, sky_g, sky_r, sky_i, sky_z, ra, dec, mjd



def compute_geometry(ra, dec, xdates, nthreads=16):
    # compute the azimuts/altitudes in parallel for the list 
    # of ra,dec's and dates

    pool = mp.Pool(nthreads)
    res = []
    for curr, curd, curx in zip(ra, dec, xdates):
        res.append(pool.apply_async(getit, (curr, curd, curx)))

    res1 = None
    for r in res:
        curv = r.get()
        if res1 is None:
            res1 = {}
            for k in curv.keys():
                res1[k] = [curv[k]]
        else:
            for k in res1.keys():
                res1[k].append(curv[k])
    pool.close()
    pool.join()
    return res1

if __name__ == '__main__':
    cached = True
    plot = True

    sky_u, sky_g, sky_r, sky_i, sky_z, ra, dec, mjd = getdata(cached)

    # evaluate dates from MJDs
    times = astropy.time.Time(mjd, format='mjd')
    xdates = [_.datetime.strftime('%Y/%m/%d %H:%M:%S') for _ in times]

    res1 = compute_geometry(ra, dec, xdates)

    for k in res1.keys():
        res1[k] = np.array(res1[k])
    for f in ('u', 'g', 'r', 'i', 'z'):
        res1['sky_%s' % f] = eval('sky_%s' % f)

    xo, yo = skymodel.getXY(objAlt=res1['objAlt'],
                            objAz=res1['objAz'], sunAlt=res1['sunAlt'], sunAz=res1['sunAz'])

    minAlt = -18 
    
    # only use objects where sun is above this altitude in the fit
    subset = (res1['sunAlt'] > minAlt)& (res1['moonP']<50)
    filts = ('u', 'g', 'r', 'i', 'z')
    
    polyTimeDeg = 2 # the degree of the time polynomial
    polyXYDeg = 2 # the degree of spatial polynomial

    def lossFunc(p, key):
        pred = skymodel.func(p, xo[subset], yo[subset], res1['sunAlt'][subset],
                             polyTimeDeg, polyXYDeg)
        delt = np.log10(res1[key])[subset] - pred
        # remove crazy outliers
        p1 = scipy.stats.scoreatpercentile(delt, 0.1)
        p2 = scipy.stats.scoreatpercentile(delt, 99.9)
        #res = np.abs(delt[betw(delt, p1, p2)]).sum()
        res = np.abs(np.clip(delt,p1, p2)).sum()
        # we use L1 norm here for robustness
        return res


    # fit the model for all the filters
    nparam = ((polyTimeDeg+1) * (polyXYDeg+1) * (polyXYDeg+2))//2
    param0 = np.zeros(nparam)

    params = [scipy.optimize.minimize(lossFunc, param0, args=('sky_%s' % _,))[
        'x'] for _ in filts]


    # evaluate the model for all the filters
    pred = {}
    models = {}
    for i, f in enumerate(filts):
        pred[f] = skymodel.func(params[i], xo, yo, res1['sunAlt'], polyTimeDeg,
                                polyXYDeg)
        np.savetxt('coeffs_%s.txt'%f,params[i])
    
    # plotting
    if plot:
        import matplotlib.pyplot as plt
        from idlplotInd import tvhist2d, plot, oplot
        plt.clf()
        minb, maxb = -0.2, 2.5
        for i, f in enumerate(filts):
            plt.subplot(1, 5, i + 1)
            if i==0:
                yt='log10(sky_{predicted})'
            else:
                yt=None
            tvhist2d(np.log10(res1['sky_%s' % f]), pred[f], minb, maxb, minb, maxb, normx='sum', ind=(res1['sunAlt'] > -20)&(res1['moonP']<50), title='%s' % f, vmax=0.3,
                     noerase=True, xtitle='log10(sky_{observed})',ytitle=yt)
            oplot([minb, maxb], [minb, maxb], color='red')
        plt.gcf().set_size_inches((10, 4))
        plt.tight_layout()

        plt.savefig('1dplots.png')
        angs = range(-20, -12)
        nangs = len(angs)
        edge = 0.8
        xgrid, ygrid = np.mgrid[-edge:edge:0.01, 0:edge:0.01]
        sh = xgrid.shape
        xgrid, ygrid = [_.flatten() for _ in [xgrid, ygrid]]
        cm = 'jet'
        bins = [30, 30]
        for ii, f in enumerate(filts):
            plt.clf()
            for i, a in enumerate(angs):
                xind = betw(res1['sunAlt'], a - 0.5, a + 0.5) & (res1['moonP']<50)
                plt.subplot(nangs, 3, i * 3 + 1)
                if i==0:
                    tit1='log10(sky) data'
                    tit2='sky residuals '
                    tit3='log10(model)'
                else:
                    tit1,tit2,tit3=[None]*3
                tvhist2d(xo, yo, -edge, edge, 0, edge, ind=xind, weights=np.log10(res1['sky_%s' % f]), statistic='median', vmin=minb, vmax=maxb, cmap=cm,
                         noerase=True, bins=bins,title=tit1)
                plt.subplot(nangs, 3, i * 3 + 2)
                tvhist2d(xo, yo, -edge, edge, 0, edge, ind=xind, weights=np.log10(res1['sky_%s' % f]) - pred[f], statistic='median', vmin=-0.5, vmax=0.5, cmap=cm,
                         noerase=True, bins=bins,title=tit2)
                plt.subplot(nangs, 3, i * 3 + 3)
                curpred = skymodel.func(
                    params[ii], xgrid, ygrid, xgrid * 0 + a, polyTimeDeg, polyXYDeg)
                tvhist2d(xgrid, ygrid, -edge, edge, 0, edge, weights=curpred, statistic='median', vmin=minb, vmax=maxb, cmap=cm, noerase=True,
                         bins=sh, title=tit3)
            plt.gcf().set_size_inches((10, 20))
            plt.tight_layout()
            plt.savefig('map_%s.png' % f)
