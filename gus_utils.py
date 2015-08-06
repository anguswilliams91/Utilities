import numpy as np 
import astropy.units as u 
from astropy.coordinates import SkyCoord


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

def galactic2cartesian(s,b,l):
	"""Return x,y,z given s,b,l"""
	return s*np.cos(b)*np.cos(l) - 8.5, s*np.cos(b)*np.sin(l), s*np.sin(b)

def helio2galactic(vLOS,l,b):
	"""Correct a heliocentric RV using the solar peculiar motion (n.b. l and b must be in radians)"""
	vLSR = 220.
	vpec = [14.0,12.24,7.25] #from Schoenrich (2012)
	return vLOS + (vpec[0]*np.cos(l) + (12.24+220.)*np.sin(l))*np.cos(b) + 7.25*np.sin(b)

