import numpy as np 
import astropy.units as u 
from astropy.coordinates import SkyCoord

abundancefile = "/home/aamw3/Dropbox/PhD_dd/python_utils/asplund_abundances.txt"
elements = np.genfromtxt(abundancefile,usecols=0,dtype=str)
solarabundances = np.genfromtxt(abundancefile,usecols=1,dtype=np.float64)

def radec2galactic(ra,dec,degrees=True):
    """convert ra, dec to galactic coords. Default is that ra,dec are in degrees"""
    if degrees:
        c = SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame='icrs')
        c = c.transform_to('galactic')
        l,b = np.array(c.l)*(np.pi/180.),np.array(c.b)*(np.pi/180.)
    else:
        c = SkyCoord(ra=ra*u.radian,dec=dec*u.radian,frame='icrs')
        c = c.transform_to('galactic')
        l,b = np.array(c.l)*(np.pi/180.),np.array(c.b)*(np.pi/180.)     
    return l,b

def galactic2radec(l,b,degrees=False):
    if degrees:
        c = SkyCoord(l=l*u.degree,b=b*u.degree,frame='galactic')
        c = c.transform_to('icrs')
        ra,dec = np.array(c.ra),np.array(c.dec)
    else:
        c = SkyCoord(l=l*u.radian,b=b*u.radian,frame='galactic')
        c = c.transform_to('icrs')
        ra,dec = np.array(c.ra),np.array(c.dec) 
    return ra,dec

def galactic2cartesian(s,b,l,Rsolar=8.5):
    """Return x,y,z given s,b,l"""
    return -s*np.cos(b)*np.cos(l) + Rsolar, -s*np.cos(b)*np.sin(l), s*np.sin(b)

def helio2galactic(vLOS,l,b,vcirc=240.,vpec= [11.1,12.24,7.25]):
    """Correct a heliocentric RV using the solar peculiar motion (n.b. l and b must be in radians)"""
    return vLOS + (vpec[0]*np.cos(l) + (vpec[1]+vcirc)*np.sin(l))*np.cos(b) + vpec[2]*np.sin(b)

def propermotion_radec2lb(pm_ras,pm_dec,ra,dec):
    """Convert proper motions pm_ras = mu_ra*cos(dec) and pm_dec = mu_dec to pm_ls = mu_l*cos(b),pm_b=mu_b"""
    raG,decG = np.radians(192.85948),np.radians(27.12825) #equatorial coords of the north galactic pole
    ra,dec = np.radians(ra),np.radians(dec)
    C1,C2 = np.sin(decG)*np.cos(dec) - np.cos(decG)*np.sin(dec)*np.cos(ra-raG),\
                    np.cos(decG)*np.sin(ra-raG)
    cosb = np.sqrt(C1**2.+C2**2.)
    pmls,pmb = cosb**-1. * (C1*pm_ras+C2*pm_dec), cosb**-1. * (-C2*pm_ras + C1*pm_dec) #do the coord transformation
    return pmls,pmb

def propermotion_lb2radec(pm_l,pm_b,ra,dec):
    """Convert proper motions pm_ras = mu_ra*cos(dec) and pm_dec = mu_dec to pm_ls = mu_l*cos(b),pm_b=mu_b"""
    raG,decG = np.radians(192.85948),np.radians(27.12825) #equatorial coords of the north galactic pole
    ra,dec = np.radians(ra),np.radians(dec)
    C1,C2 = np.sin(decG)*np.cos(dec) - np.cos(decG)*np.sin(dec)*np.cos(ra-raG),\
                    np.cos(decG)*np.sin(ra-raG)
    cosb = np.sqrt(C1**2.+C2**2.)
    pm_ra,pm_dec = cosb**-1. * (C1*pm_l-C2*pm_b), cosb**-1. * (C2*pm_l + C1*pm_b) #do the coord transformation
    return pm_ra,pm_dec

def obs2cartesian(pm1,pm2,ra,dec,s,vhelio,radec_pms=False,Rsolar=8.5,Schoenrich=True):
    """Convert observed stuff to cartesian velocities (ra,dec must be in degrees, s in kpc and vhelio in kms-1)
    The coordinate transformations are according to the conventions found in Bond et al. (2010)"""
    if radec_pms: pml,pmb = propermotion_radec2lb(pm1,pm2,ra,dec)
    else: pml,pmb = pm1,pm2
    vl,vb = 4.74*s*pml,4.74*s*pmb #tangential motions
    l,b = radec2galactic(ra,dec)
    #now calculate the (uncorrected for solar motion) cartesian velocities
    vx  = -vhelio*np.cos(l)*np.cos(b) + vb*np.cos(l)*np.sin(b) + vl*np.sin(l)
    vy = -vhelio*np.sin(l)*np.cos(b) + vb*np.sin(l)*np.sin(b) - vl*np.cos(l)
    vz = vhelio*np.sin(b) + vb*np.cos(b)
    #correct for solar motion
    if not Schoenrich:
        vLSR,vXsun,vYsun,vZsun = -220.,-10.,-5.3,7.2
    else:
        #use updated LSR from Schoenrich and Bovy's circular speed
        vLSR,vXsun,vYsun,vZsun = -240., -11.1, -12.24, 7.25
    vx,vy,vz = vx + vXsun,vy+vYsun+vLSR,vz+vZsun
    #now convert to cylindrical coords
    x,y,z = galactic2cartesian(s,b,l,Rsolar=Rsolar)
    return x,y,z,vx,vy,vz 

def cartesian2cylindrical(x,y,z,vx,vy,vz):
    """Convert cartesian positions and velocities into cylindrical ones"""
    R = np.sqrt(x**2.+y**2.)
    phi = np.arctan2(y,x)
    vR = (vx*x+vy*y)/R
    vphi = (-vx*y+vy*x)/R
    return R,z,phi,vR,vz,vphi

def cartesian2spherical(x,y,z,vx,vy,vz):
    """Convert cartesian positions and velocities into spherical ones"""
    r = np.sqrt(x**2.+y**2.+z**2.)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    st,ct,sp,cp = np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
    vr,vtheta,vphi = vx*st*cp+vy*st*sp+vz*ct,vx*cp*ct+vy*sp*ct-vz*st,-vx*sp+vy*cp
    return r,theta,phi,vr,vtheta,vphi

def cylindrical2spherical(R,z,vR,vz):
    """Convert cylindrical velocities to spherical ones"""
    r=np.sqrt(R**2.+z**2.)
    return vR*(R/r) + vz*(z/r), vR*(z/r) - vz*(R/r) #vr,vtheta

def cylindrical2spheroidal(delta,R,z,vR,vz):
    """Compute the stackel velocities given cylindrical velocities and positions"""
    #Transform R and z to u and v
    d12= (z+delta)**2.+R**2.
    d22= (z-delta)**2.+R**2.
    coshu= 0.5/delta*(np.sqrt(d12)+np.sqrt(d22))
    cosv=  0.5/delta*(np.sqrt(d12)-np.sqrt(d22))
    u= np.arccosh(coshu)
    v= np.arccos(cosv)
    
    #now calculate the velocities
    vu = (vR*np.cosh(u)*np.sin(v) + vz*np.sinh(u)*np.cos(v))/np.sqrt(np.sinh(u)**2.+np.sin(v)**2.)
    vv = (vR*np.sinh(u)*np.cos(v) - vz*np.cosh(u)*np.sin(v))/np.sqrt(np.sinh(u)**2.+np.sin(v)**2.)

    return u,v,vu,vv

def spherical2cartesian(r,theta,phi,vr,vtheta,vphi):
    x,y,z = r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)
    vx = vr*np.sin(theta)*np.cos(phi) + vtheta*np.cos(theta)*np.cos(phi) - vphi*np.sin(phi)
    vy = vr*np.sin(theta)*np.sin(phi) + vtheta*np.cos(theta)*np.sin(phi) + vphi*np.cos(phi)
    vz = vr*np.cos(theta) - vtheta*np.sin(theta)

    return x,y,z,vx,vy,vz

def cartesian2observable(x,y,z,vx,vy,vz,Rsolar=8.5,Schoenrich=True):
    """Transform cartesian coords to (s,l,b,vLOS,mul,mub)"""
    s = np.sqrt((x-Rsolar)**2.+y**2.+z**2.)
    b = np.arcsin(z/s)
    l = np.arctan2(y,(x-Rsolar))+np.pi
    if not Schoenrich:
        vLSR,vXsun,vYsun,vZsun = -220.,-10.,-5.3,7.2
    else:
        #use updated LSR from Schoenrich and Bovy's circular speed
        vLSR,vXsun,vYsun,vZsun = -240., -11.1, -12.24, 7.25
    vx,vy,vz = vx - vXsun,vy-vYsun-vLSR,vz-vZsun #transform to frame where we haven't corrected for solar motion
    sb,cb,sl,cl = np.sin(b),np.cos(b),np.sin(l),np.cos(l)
    vLOS = vz*sb-cb*(vx*cl+vy*sl)
    vl = -vy*cl+vx*sl
    vb = cb*(vz+vx*cl*(sb/cb)+vy*sl*(sb/cb))
    pml,pmb = vl/(4.74*s),vb/(4.74*s)
    ra,dec = galactic2radec(l,b)
    return pml,pmb,ra,dec,s,vLOS

def correct_abundance(abun,feh,elemstr):
    """Correct an abundance to relative to solar, from a log epsilon measurement, using asplund 2009
        returns [Elem / Fe]"""
    if elemstr in elements:
        solarabun = np.float(solarabundances[np.where(elements==elemstr)])
        return abun - feh - solarabun
    else:
        print "Unrecognised element!"
        return None

def radec2sag(ra,dec,indegrees=True,outdegrees=True):
    """Coordinates from Vasily's 2014 paper for sag (tilda coords from appendix A)"""
    if indegrees: ra,dec = np.radians(ra),np.radians(dec)
    lambdat = np.arctan2(-0.93595354*np.cos(ra)*np.cos(dec) - 0.31910658*np.sin(ra)*np.cos(dec) + \
                            0.14886895*np.sin(dec), 0.21215555*np.cos(ra)*np.cos(dec) - \
                            0.84846291*np.sin(ra)*np.cos(dec) - 0.48487186*np.sin(dec))
    Bt = np.arcsin(0.28103559*np.cos(ra)*np.cos(dec) - 0.42223415*np.sin(ra)*np.cos(dec) + 0.86182209*np.sin(dec))
    if outdegrees: return np.degrees(lambdat),np.degrees(Bt)
    else: return lambdat, Bt

def sag2radec(lambdat,Bt,indegrees=False):
    """Go the other way to the above function"""
    if indegrees: lambdat,Bt = np.radians(lambdat),np.radians(Bt)
    ra = np.arctan2(-0.84846291*np.cos(lambdat)*np.cos(Bt)-0.31910658*np.sin(lambdat)*np.cos(Bt)-0.42223415*np.sin(Bt),\
                        0.21215555*np.cos(lambdat)*np.cos(Bt) - 0.93595354*np.sin(lambdat)*np.cos(Bt)+0.28103559*np.sin(Bt))
    dec = np.arcsin(-0.48487186*np.cos(lambdat)*np.cos(Bt)+0.14886895*np.sin(lambdat)*np.cos(Bt)+0.86182209*np.sin(Bt))
    return np.degrees(ra),np.degrees(dec)

def angular_radius_selection(ra,dec,pos,radius):
    """Grab everything within a specific angular radius on the sky [ra0,dec0] is the centre of the 
    circle and radius is the angular radius of the region in degrees. Assumes everything is given in 
    degrees."""
    ra0,dec0 = pos
    ra0,dec0 = np.radians(ra0),np.radians(dec0)
    idx = np.sin(dec0)*np.sin(np.radians(dec)) + np.cos(dec0)*np.cos(np.radians(dec))*np.cos(ra0-np.radians(ra)) > np.cos(np.pi*radius/180.)
    return ra[idx],dec[idx],idx

def BHB_distance(g,r):
    """given the SDSS g and r band magnitudes of a BHB, compute its distance using the relation from Deason et al. (2011)"""
    G =  0.434 - 0.169*(g-r) +2.319*(g-r)**2. + 20.449*(g-r)**3. + 94.517*(g-r)**4.
    mu = g-G
    return (10.**(1.+.2*mu))/1000. #distance in kpc

"""Stuff for emcee"""

def write_to_file(sampler,outfile,p0,Nsteps=10):
    """Quick thing for writing a chain to file"""
    f = open(outfile,"a")
    f.close()
    for result in sampler.sample(p0,iterations=Nsteps,storechain=False):
        position = result[0]
        f = open(outfile,"a")
        for k in range(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,position[k]))))
        f.close()
    return None

def reshape_chain(chain):
    #take numpy array of shape (nwalkers*nsteps,ndim+1) and reshape to (nwalkers,nsteps,ndim)
    nwalkers = len(np.unique(chain[:,0]))
    nsteps = len(chain[:,0])/nwalkers
    ndim = np.shape(chain)[1]-1
    c = np.zeros((nwalkers,nsteps,ndim))
    #put the chains into the right shape
    for i in np.arange(nwalkers):
        idx = chain[:,0]==i
        for j in np.arange(1,ndim+1):
            c[i,:,j-1] = chain[idx,j]
    return c

def GelmanRubin(chain,burnin=None):
    """Perform the Gelman-Rubin test for convergence"""
    c = reshape_chain(chain)
    if burnin is not None:
        c=c[:,burnin:,:]
    nwalkers,nsteps,ndim = np.shape(c)
    gr = np.zeros(ndim)
    for j in np.arange(ndim):
        a = np.mean(np.var(c[:,:,j],axis=1))
        b = np.var(np.mean(c[:,:,j],axis=1))
        va = (1.-np.float(nsteps)**-1.)*a + (1./np.float(nsteps))*b
        gr[j]=np.sqrt(va/a)
    return gr

def chain_results(chain,burnin=None):
    """Get the results from a chain using the 16th, 50th and 84th percentiles. 
    For each parameter a tuple is returned (best_fit, +err, -err)"""
    if burnin:
        nwalkers,nsteps,ndim = np.shape(reshape_chain(chain))
        chain = chain[nwalkers*burnin:,:]
    nwalkers,nsteps,ndim = np.shape(reshape_chain(chain))
    chain = chain[nwalkers*burnin:,:]
    return map(lambda v: [v[1],v[2]-v[1],v[1]-v[0]],\
                zip(*np.percentile(chain[:,1:],[16,50,84],axis=0)))


class FuncWrapper(object):
    #wrap functions with this so that they are pickleable
    def __init__(self, f, args=[], kwargs={}):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)

