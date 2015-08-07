import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq,minimize
from scipy.integrate import fixed_quad,quad


"""This stuff will find the actions in a general axisymmetric potential using the stackel fudge of Binney (2012)"""

BRENTERROR= 123000.

def uv2Rz(u,v,delta):
	"""Transform from confocal ellipsoidal coords to cylindrical coords"""
	return (delta*np.sinh(u)*np.sin(v), delta*np.cosh(u)*np.cos(v))

def Rz2uv(R,z,delta):
    """Transform R and z to u and v"""
    d12= (z+delta)**2.+R**2.
    d22= (z-delta)**2.+R**2.
    coshu= 0.5/delta*(np.sqrt(d12)+np.sqrt(d22))
    cosv=  0.5/delta*(np.sqrt(d12)-np.sqrt(d22))
    u= np.arccosh(coshu)
    v= np.arccos(cosv)
    return (u,v)

def dU(u,u0,v,delta,Phi):
	"""The first of Equations (8) from Binney (2012), required to compute the approx. integral of motion for Ju"""
	R,z = uv2Rz(u,v,delta)
	R0,z0 = uv2Rz(u0,v,delta)
	return (np.sinh(u)**2. + np.sin(v)**2.)*Phi(R,z) - (np.sinh(u0)**2. + np.sin(v)**2.)*Phi(R0,z0) 

def dV(u,v,delta,Phi):
	"""The second of Equations (8) from Binney (2012), required to compute the approx. integral of motion for Jv"""
	R,z = uv2Rz(u,v,delta)
	R0,z0 = uv2Rz(u,np.pi/2.,delta)
	return Phi(R0,z0)*np.cosh(u)**2. - (np.sinh(u)**2. + np.sin(v)**2.)*Phi(R,z)

def deriv_dU(u,v,delta,Phi):
	"""The derivative of dU w.r.t. u, so that we can minimize this function w.r.t. u and choose an appropriate u0"""
	R,z = uv2Rz(u,v,delta)
	dRdu = delta*np.cosh(u)*np.sin(v)
	dzdu = delta*np.sinh(u)*np.cos(v)
	dPhidu = Phi.dR(R,z)*dRdu + Phi.dz(R,z)*dzdu
	return (np.sinh(u)**2. + np.sin(v)**2.)*dPhidu + 2.*np.sinh(u)*np.cosh(u)*Phi(R,z)

def get_consts(orbit,Phi):
	"""Given an orbit and a potential (and a focal distance), derive the exact and approximate integrals of motion"""
	R,z,vR,vz,vphi = orbit
	delta = find_delta(R,z,Phi) #get a decent value for delta
	uorb, vorb = Rz2uv(R,z,delta) #get the current (u,v) coords of this orbit
	u0 = uorb #for single orbits it doesn't matter what this is
	E = .5*(vR**2. + vz**2. + vphi**2.) + Phi(R,z) #energy of this orbit
	Jphi = R*vphi #angular momentum about the z-axis is an action
	#calculate the canonical momenta at the phase space point we possess
	pu0 = delta*(vR*np.cosh(uorb)*np.sin(vorb) + vz*np.sinh(uorb)*np.cos(vorb))
	pv0 = delta*(vR*np.sinh(uorb)*np.cos(vorb) - vz*np.cosh(uorb)*np.sin(vorb))
	#now the approx integrals of motion
	I3U = E*np.sinh(uorb)**2. - pu0**2./(2.*delta**2.) - Jphi**2./(2.*delta**2.*np.sinh(uorb)**2.)-dU(uorb,u0,vorb,delta,Phi)
	I3V = -E*np.sin(vorb)**2. + pv0**2./(2.*delta**2.) + Jphi**2./(2.*delta**2.*np.sin(vorb)**2.)-dV(uorb,vorb,delta,Phi)
	return uorb,vorb,u0,E,Jphi,I3U,I3V,delta

def pu_squared(u,u0,vorb,E,Jphi,I3U,Phi,delta):
	"""Equation for pu^2 / 2*delta^2"""
	return E*np.sinh(u)**2. - (I3U+dU(u,u0,vorb,delta,Phi)) - Jphi**2./(2.*delta**2.*np.sinh(u)**2.)

def pv_squared(v,uorb,E,Jphi,I3V,Phi,delta):
	"""Equation for pv^2 / 2*delta^2"""
	return E*np.sin(v)**2. + I3V+dV(uorb,v,delta,Phi) - Jphi**2./(2.*delta**2.*np.sin(v)**2.)

def Ju_integrand(u,u0,vorb,E,Jphi,I3U,Phi,delta):
	"""The integrand for Ju"""
	return np.sqrt(pu_squared(u,u0,vorb,E,Jphi,I3U,Phi,delta)*2.)*delta*(1./np.pi)

def Jv_integrand(v,uorb,E,Jphi,I3V,Phi,delta):
	"""The integrand for Jv"""
	return np.sqrt(pv_squared(v,uorb,E,Jphi,I3V,Phi,delta)*2.)*delta*(2./np.pi)

def find_u0(uorb,vorb,Phi,delta):
	"""Find the optimal value for u0 using nelder-mead, only necessary if one wishes to grid the actions"""
	res = minimize(deriv_dU,uorb,(vorb,delta,Phi))
	if res.success:
		return res.x
	elif deriv_dU(res.x,vorb,delta,Phi)<1e-3:
		return res.x #acceptable
	else:
		return uorb #can't be bothered doing something clever yet

def find_umin_umax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta):
	"""Find the limits of integration for the radial action"""
	eps = 1e-8
	if np.abs(pu_squared(uorb,u0,vorb,E,Jphi,I3U,Phi,delta)) < 1e-7: #in this case we are at umin or umax
		#discover which one
		peps = pu_squared(uorb+eps,u0,vorb,E,Jphi,I3U,Phi,delta)
		meps = pu_squared(uorb-eps,u0,vorb,E,Jphi,I3U,Phi,delta)
		if peps<0. and meps>0.:
			#we are at umax
			umax = uorb
			rstart = findstart_uminumax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta)
			if rstart==0.: umin=0.
			else:
				try:
					umin = brentq(pu_squared,rstart,umax-eps,(u0,vorb,E,Jphi,I3U,Phi,delta),maxiter=200)
				except ValueError:
					print "This orbit has either zero or extremely small binding energy. Can't compute Ju"
					return (BRENTERROR,BRENTERROR)
		elif peps>0. and meps<0.:
			#we're at umin
			rend = findstart_uminumax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta,umax=True)
			umax = brentq(pu_squared,uorb+eps,rend,(u0,vorb,E,Jphi,I3U,Phi,delta),maxiter=200)
		else:
			umin = uorb
			umax = uorb #circular orbit
	else:
		rstart = findstart_uminumax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta)
		if rstart==0.: umin=0.
		else:
			try:
				umin = brentq(pu_squared,rstart,rstart/0.9,(u0,vorb,E,Jphi,I3U,Phi,delta),maxiter=200)
			except ValueError:
				print "This orbit has either zero or extremely small binding energy. Can't compute Ju"
				return (BRENTERROR,BRENTERROR)
		rend = findstart_uminumax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta,umax=True)
		umax = brentq(pu_squared,rend/1.1,rend,(u0,vorb,E,Jphi,I3U,Phi,delta),maxiter=200)
	return (umin,umax)

def find_vmin(vorb,uorb,E,Jphi,I3V,Phi,delta):
	"""Find the lower limit of integration for the vertical action"""
	eps=1e-8
	if np.abs(pv_squared(vorb,uorb,E,Jphi,I3V,Phi,delta))<1e-7:
		#we are at either vmin or vmax
		peps = pv_squared(vorb+eps,uorb,E,Jphi,I3V,Phi,delta)
		meps = pv_squared(vorb-eps,uorb,E,Jphi,I3V,Phi,delta)
		if peps<0. and meps>0.:
			#we're at vmax
			rstart = findstart_vmin(vorb,uorb,E,Jphi,I3V,Phi,delta)
			if rstart==0.: return 0.
			else:
				try:
					return brentq(pv_squared,rstart,vorb-eps,(uorb,E,Jphi,I3V,Phi,delta),maxiter=200)
				except ValueError:
					print "This orbit has either zero or very small binding energy. Can't compute the vertical action."
					return BRENTERROR
		elif peps>0. and meps<0.:
			#we're at vmin
			return vorb
		else:
			#orbit is in the equatorial plane
			return vorb
	else:
		rstart = findstart_vmin(vorb,uorb,E,Jphi,I3V,Phi,delta)
		if rstart==0.: return 0.
		else:		
			try:
				return brentq(pv_squared,rstart,vorb-eps,(uorb,E,Jphi,I3V,Phi,delta),maxiter=200)
			except ValueError:
				print "This orbit either has either zero or very small binding energy. Can't compute the vertical action."
	return None

def findstart_vmin(vorb,uorb,E,Jphi,I3V,Phi,delta):
	"""Find a suitable starting point for using Brent's method for finding the minimum v coordinate"""
	vtry = 0.9*vorb
	while pv_squared(vtry,uorb,E,Jphi,I3V,Phi,delta)>=0. and vtry>1e-9:
		vtry*=0.9
	if vtry<1e-9:
		return 0.
	return vtry

def findstart_uminumax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta,umax=False):
	"""Find suitable starting points for using Brent's method to find the limits of the Ju integral"""
	if umax:
		utry = 1.1*uorb
	else:
		utry = 0.9*uorb
	while pu_squared(utry,u0,vorb,E,Jphi,I3U,Phi,delta)>=0. and utry>1e-9:
		if umax:
			if utry>1000.:
				print "This orbit looks unbound."
				return BRENTERROR
			utry*=1.1
		else:
			utry*=0.9
	if utry<1e-9: return 0.
	return utry

def get_actions_stackelfudge(orbit,Phi,fixed_gauss=True):
	"""Obtain the actions for an orbit (R,z,vR,vz,vphi) in the potential Phi(R,z)"""
	uorb,vorb,u0,E,Jphi,I3U,I3V,delta = get_consts(orbit,Phi)
	umin,umax = find_umin_umax(uorb,u0,vorb,E,Jphi,I3U,Phi,delta)
	vmin = find_vmin(vorb,uorb,E,Jphi,I3V,Phi,delta)
	if fixed_gauss:
		Ju = fixed_quad(Ju_integrand,umin,umax,(u0,vorb,E,Jphi,I3U,Phi,delta),n=11)[0]
		Jv = fixed_quad(Jv_integrand,vmin,np.pi/2,(uorb,E,Jphi,I3V,Phi,delta),n=11,)[0]
	else:
		Ju = quad(Ju_integrand,umin,umax,(u0,vorb,E,Jphi,I3U,Phi,delta))[0]
		Jv = quad(Jv_integrand,vmin,np.pi/2,(uorb,E,Jphi,I3V,Phi,delta))[0]
	return Ju,Jv,Jphi

def find_delta(R,z,Phi):
	"""Find a suitable value of delta for a given orbit"""
	delta2= (z**2.-R**2. #eqn. (9) has a sign error
                 +(-3.*R*Phi.dz(R,z)
                   +3.*z*Phi.dR(R,z)
                   +R*z*(Phi.dRR(R,z)
                         -Phi.dzz(R,z)))/Phi.dRz(R,z))
	if delta2 < 0. and delta2 > -10.**-10.: delta2= 0.
	return np.sqrt(delta2)


"""This stuff will find the actions in a spherical potential"""

def rperi_rapo_eq(r,E,L,Phi):
	"""The equation for finding peri and apo""" 
	return 2.*E - 2.*Phi(r,0.) - L**2. / r**2.

def jr_integrand(r,E,L,Phi):
	return np.sqrt(rperi_rapo_eq(r,E,L,Phi))

def rp_ra_getstart(r,E,L,Phi,apo=False):
	"""Find suitable place to start the minimization routine"""
	if apo:
		rtry = 1.1*r
	else:
		rtry = 0.9*r
	while rperi_rapo_eq(rtry,E,L,Phi)>=0. and rtry>1e-9:
		if apo:
			if rtry>1000.:
				print "This orbit looks unbound."
				return BRENTERROR
			rtry*=1.1
		else:
			rtry*=0.9
	if rtry<1e-9:
		return 0.
	return rtry

def get_rp_ra(r,E,L,Phi):
	"""Find the limits for the radial action integral"""
	eps=1e-8
	if np.abs(rperi_rapo_eq(r,E,L,Phi))<1e-7: #we are at peri or apo
		peps = rperi_rapo_eq(r+eps,E,L,Phi)
		meps = rperi_rapo_eq(r-eps,E,L,Phi)
		if peps<0. and meps>0.: #we are at apo
			ra = r
			rstart = rp_ra_getstart(r-eps,E,L,Phi)
			if rstart==0.: rp = 0.
			else:
				try:
					rp = brentq(rperi_rapo_eq,rstart,r-eps,(E,L,Phi),maxiter=200)
				except ValueError:
					print "This orbit has either zero or extremely small binding energy. Can't find Jr"
					return BRENTERROR
		elif peps>0. and meps<0.: #we are at peri
			rp = r
			rend = rp_ra_getstart(r,E,L,Phi,apo=True)
			ra = brentq(rperi_rapo_eq,r+eps,rend,rend)
		else: #circular orbit
			rp = r
			ra = r
	else:
		rstart = rp_ra_getstart(r,E,L,Phi)
		if rstart==0.: rp=0.
		else:
			try:
				rp = brentq(rperi_rapo_eq,rstart,rstart/0.9,(E,L,Phi),maxiter=200)
			except ValueError:
				"This orbit has either zero or extremely small binding energy. Can't find Jr."
				return BRENTERROR
		rend = rp_ra_getstart(r,E,L,Phi,apo=True)
		ra = brentq(rperi_rapo_eq,rend/1.1,rend,(E,L,Phi),maxiter=200)
	return (rp,ra)

def get_actions_spherical(orbit,Phi,fixed_gauss=True):
	if len(orbit)==3.:
		#just give a radius, energy and angular momentum
		r,E,L = orbit
	elif len(orbit)==5.:
		#give full 6D
		r,theta,vr,vtheta,vphi = orbit
		E = .5*(vr**2.+vtheta**2.+vphi**2.)+Phi(r,0.)
		L = r*np.sqrt(vtheta**2.+vphi**2.)
		Jphi = r*np.sin(theta)*vphi
	rp,ra = get_rp_ra(r,E,L,Phi)
	if rp==BRENTERROR or ra==BRENTERROR:
		return 0.
	if fixed_gauss:
		Jr = (1./np.pi)*fixed_quad(jr_integrand,rp,ra,(E,L,Phi),n=11)[0]
	else:
		Jr = (1./np.pi)*quad(jr_integrand,rp,ra,(E,L,Phi))[0]
	if len(orbit)==3.:
		#just return the angular momentum and the radial action
		return (L,Jr)
	elif len(orbit)==5.:
		return (Jphi,L-np.abs(Jphi),Jr)
