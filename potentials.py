import numpy as np
from actions import *
from galpy.potential import MWPotential2014, evaluatePotentials, evaluateRforces,evaluatezforces,evaluateR2derivs,evaluatez2derivs,evaluateRzderivs

_G = 429960.6896 #Gravitational constant when speeds are measured in kms**-1, masses in 10**11 M_solar and distances in kpc. 

class LogHalo():

	"""A flattened logarithmic halo potential"""

	def __init__(self,v0=220.,q=1.,Rc=1e-8):
		self.v0 = v0
		self.q = q
		self.Rc = Rc
	
	def __call__(self,R,z):
		"""Evaluate the potential"""
		return (self.v0**2. / 2.) * np.log(self.Rc**2. + R**2. + (z/self.q)**2.)

	def dR(self,R,z):
		"""Evalute the derivative of Phi w.r.t. R"""
		return self.v0**2. * R / (self.Rc**2. + R**2. + (z/self.q)**2.)

	def dz(self,R,z):
		"""Evalute the derivative of Phi w.r.t z"""
		return self.v0**2. * (z/self.q**2.) / (self.Rc**2. + R**2. + (z/self.q)**2.)

	def dRR(self,R,z):
		"""Second derivative R,R"""
		return self.v0**2. / (self.Rc**2. + R**2. + (z/self.q)**2.) - self.v0**2. * 2.*  R * R / (self.Rc**2. + R**2. + (z/self.q)**2.)**2.

	def dzz(self,R,z):
		"""Second derivative z,z"""
		return self.v0**2. * (1./self.q**2.) / (self.Rc**2. + R**2. + (z/self.q)**2.) - self.v0**2.* 2. * (z**2./self.q**4.) / (self.Rc**2. + R**2. + (z/self.q)**2.)**2.

	def dRz(self,R,z):
		"""Second derivative R,z"""
		return -self.v0**2. * R * (2.*z/self.q**2.) / (self.Rc**2. + R**2. + (z/self.q)**2.)**2.

	def circspeed(self,R):
		return np.sqrt(R * self.dR(R,0.))

	def findJs(self,orbit,fixed_gauss=True):
		"""Find the actions for a given set of initial conditions"""
		if self.q!=1.:
			return get_actions_stackelfudge(orbit,self,fixed_gauss=fixed_gauss)
		else:
			return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)

class MilkyWay(LogHalo):

	"""Wrapper for the Galpy MWPotential2014"""

	def __init__(self):
		self.pot = MWPotential2014

	def __call__(self,R,z):
		return evaluatePotentials(R,z,self.pot)

	def dR(self,R,z):
		return -evaluateRforces(R,z,self.pot)

	def dz(self,R,z):
		return -evaluatezforces(R,z,self.pot)

	def dRR(self,R,z):
		return evaluateR2derivs(R,z,self.pot)

	def dzz(self,R,z):
		return evaluatez2derivs(R,z,self.pot)

	def dRz(self,R,z):
		return evaluateRzderivs(R,z,self.pot)

	def circspeed(self,R):
		return np.sqrt(R*self.dR(R,0.))

	def findJs(self,orbit,fixed_gauss=True):
		return get_actions_stackelfudge(orbit,self,fixed_gauss=fixed_gauss)

class PowerLawPotential(LogHalo):

	"""A power law model (can be flattened)"""

	def __init__(self,v0=220.,q=1.,gamma=0.5):
		self.v0=v0
		self.q=q
		self.gamma=gamma


	def __call__(self,R,z):
		rq = np.sqrt(R**2. + (z/self.q)**2.)
		return (self.v0**2./self.gamma)*rq**self.gamma

	def dR(self,R,z):
		rq = np.sqrt(R**2. + (z/self.q)**2.)
		return R*(self.v0**2.)*rq**(self.gamma - 2.)

	def dz(self,R,z):
		rq = np.sqrt(R**2.+(z/self.q)**2.)
		return (z/self.q**2.)*self.v0**2. * rq**(self.gamma - 2.)

	def dRR(self,R,z):
		rq = np.sqrt(R**2.+(z/self.q)**2.)
		return self.v0**2.*rq**(self.gamma-2.)+R**2.*self.v0**2.*(self.gamma-2.)*rq**(self.gamma-4.)

	def dzz(self,R,z):
		rq = np.sqrt(R**2. + (z/self.q)**2.)
		return self.q**-2.*self.v0**2.*rq**(self.gamma-2.)+(z/self.q**2.)**2.*self.v0**2.*(self.gamma-2.)*rq**(self.gamma-4.)

	def dRz(self,R,z):
		rq = np.sqrt(R**2. + (z/self.q)**2.)
		return (R*z/self.q**2.)*(self.gamma-2.)*self.v0**2.*rq**(self.gamma-4.)

	def circspeed(self,R):
		return np.sqrt(R*self.dR(R,0.))


class MiyamotoNagai(LogHalo):

	"""A Miyamoto-Nagai potential"""

	def __init__(self,M=1.,a=1.,b=1.):
		self.M = M
		self.a = a
		self.b = b

	def __call__(self,R,z):
		return -_G*self.M*(R**2. + (self.a + np.sqrt(z**2. + self.b**2.))**2.)**-.5

	def dR(self,R,z):
		return _G*self.M*R*(R**2. + (self.a + np.sqrt(self.b**2.  + z**2.))**2.)**-1.5

	def dz(self,R,z):
		c = self.a + np.sqrt(self.b**2.+z**2.)
		return _G*self.M*z*c * (self.b**2.+z**2.)**-.5 * (R**2. + c**2.)**-1.5

	def dRR(self,R,z):
		return _G*self.M*(self.a**2. + self.b**2. - 2.*R**2. + z**2. + 2.*self.a*np.sqrt(self.b**2.+z**2.))*(R**2.+(self.a**2.+np.sqrt(self.b**2.+z**2.))**2.)**-2.5

	def dzz(self,R,z):
		return (_G*self.M*(self.a**3.*self.b**2.+self.a**2.*(3.*self.b**2.-2.*z**2.)*np.sqrt(self.b**2.+z**2.)\
						+(self.b**2.+R**2.-2.*z**2.)*(self.b**2.+z**2.)**1.5+self.a*(3.*self.b**4.-4.*z**4.+self.b**2.*(R-z)*(R+z))))/((self.b**2.+z**2.)**1.5*\
																																	(R**2.+(self.a+np.sqrt(self.b**2.+z**2.))**2.)**2.5)

	def dRz(self,R,z):
		c=self.a+np.sqrt(self.b**2.+z**2.)
		return -3.*_G*self.M*R*z*c*((c-self.a)*(R**2. + c**2.)**2.5)


	def findJs(self,orbit,fixed_gauss=True):
		"""Find the actions for a given set of initial conditions"""
		if self.a==0.:
			return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)
		else:
			return get_actions_stackelfudge(orbit,self,fixed_gauss=fixed_gauss)

class Plummer(MiyamotoNagai):

	"""Plummer potential (just use the a=0 limit of the Miyamoto Nagai model"""

	def __init__(self,M=1.,b=1.):
		self.M=M
		self.a=0.
		self.b=b

	def findJs(self,orbit,fixed_gauss=True):
		return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)

class NFW(LogHalo):

	"""NFW halo (default params are from Piffl et. al (2014)"""

	def __init__(self,rho0=0.0001816,rs=14.4):
		self.rho0=rho0
		self.rs=rs

	def __call__(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		return -4.*np.pi*_G*self.rs**2.*np.log(1.+r/self.rs)*self.rs/r

	def dR(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		drdR = R/r
		return -4.*np.pi*_G*self.rho0*self.rs**3.*(r - (self.rs+r)*np.log(1.+r/self.rs))/(R**2.*(self.rs+r))*drdR

	def dz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		drdz = z/r
		return -4.*np.pi*_G*self.rho0*self.rs**3.*(r - (self.rs+r)*np.log(1.+r/self.rs))/(R**2.*(self.rs+r))*drdz

	def dRR(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		d2phidr2 = -4.*np.pi*_G*self.rs**3.*self.rho0**2.*(r*(3.*r+2.*self.rs)-2.*(r+self.rs)**2.*np.log(1.+r/self.rs))/\
														(r**3.*(r+self.rs)**2.)
		return self.dz(R,z)*(z/r**2.) + (R**2./r**2.)*d2phidr2

	def dzz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		d2phidr2 = -4.*np.pi*_G*self.rs**3.*self.rho0**2.*(r*(3.*r+2.*self.rs)-2.*(r+self.rs)**2.*np.log(1.+r/self.rs))/\
														(r**3.*(r+self.rs)**2.)
		return self.dR(R,z)*(R/r**2.) + (z**2./r**2.)*d2phidr2

	def dRz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		d2phidr2 = -4.*np.pi*_G*self.rs**3.*self.rho0**2.*(r*(3.*r+2.*self.rs)-2.*(r+self.rs)**2.*np.log(1.+r/self.rs))/\
														(r**3.*(r+self.rs)**2.)
		return -self.dR(R,z)*(z/r**2.) + R*z*d2phidr2/r**2.

	def findJs(self,orbit,fixed_gauss=True): 
		return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)


class Hernquist(NFW):

	"""The Hernquist sphere"""

	def __init__(self,M=1.,a=1.):
		self.a = a
		self.M = M

	def __call__(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		return -_G*self.M/(r+self.a)

	def dR(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		drdR = R/r
		return drdR*_G*self.M/(r+self.a)**2.
	
	def dz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		drdz = z/r
		return drdz*_G*self.M/(r+self.a)**2.

	def dRR(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		return (z**2./r**3.)*_G*self.M/(r+self.a)**2. - 2.*(R**2./r**2.)*_G*self.M/(r+self.a)**3.

	def dzz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		return (R**2./r**3.)*_G*self.M/(r+self.a)**2. - 2.*(z**2./r**2.)*_G*self.M/(r+self.a)**3.

	def dRz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		return R*z*(-_G*self.M*r**-3./(r+self.a)**2. + 2.*_G*self.M*r**-2./(r+self.a)**3.)

	def findJs(self,orbit,fixed_gauss=True):
		return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)


class CompositePotential(LogHalo):

	"""Class that allows for composite potentials to be created"""

	def __init__(self,potlist=[Hernquist(M=10.**-1.,a=0.7),MiyamotoNagai(M=1.,a=6.5,b=0.3),NFW()]):
		self.potlist = potlist #pass the init function a list of other potentials

	def __call__(self,R,z):
		return np.sum(map(lambda f: f(R,z), self.potlist),axis=0)

	def dR(self,R,z):
		return np.sum(map(lambda f: f.dR(R,z), self.potlist),axis=0)

	def dz(self,R,z):
		return np.sum(map(lambda f: f.dz(R,z), self.potlist),axis=0)

	def dRR(self,R,z):
		return np.sum(map(lambda f: f.dRR(R,z), self.potlist),axis=0)

	def dzz(self,R,z):
		return np.sum(map(lambda f: f.dzz(R,z), self.potlist),axis=0)

	def dRz(self,R,z):
		return np.sum(map(lambda f: f.dRz(R,z), self.potlist),axis=0)

	def findJs(self,orbit,fixed_gauss=True):
		return get_actions_stackelfudge(orbit,self,fixed_gauss=fixed_gauss)		


class Isochrone(LogHalo):

	def __init__(self,M=1.,b=1.):
		self.M = M
		self.b = b

	def __call__(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		return -_G*self.M/(self.b + np.sqrt(r**2. + self.b**2.))

	def dR(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		c = np.sqrt(self.b**2. + r**2.)
		return _G*self.M*R*(c*(c+self.b)**2.)

	def dz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		c = np.sqrt(self.b**2. + r**2.)
		return _G*self.M*z*(c*(c+self.b)**2.)

	def dRR(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		c = np.sqrt(self.b**2. + r**2.)
		return _G*self.M*(self.b**3. + self.b*z**2. +self.b**2.*c + (z**2. - 2.*R**2.)*c)/\
									(c**3.*(c+self.b)**3.)

	def dzz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		c = np.sqrt(self.b**2. + r**2.)
		return _G*self.M*(self.b**3. + self.b*R**2. +self.b**2.*c + (R**2. - 2.*z**2.)*c)/\
									(c**3.*(c+self.b)**3.)

	def dRz(self,R,z):
		r = np.sqrt(R**2.+z**2.)
		c = np.sqrt(self.b**2. + r**2.)
		return _G*self.M*R*z*(self.b+3.*c)/(c**3.*(self.b+c)**3.)

	def analyticJs(self,orbit):
		if len(orbit)==3:
			r,E,L = orbit
			Jr = _G*self.M/np.sqrt(-2.*E) - .5*(L+np.sqrt(L**2.+4.*_G*self.M*self.b))
			return L,Jr
		elif len(orbit)==5.:
			r,theta,vr,vtheta,vphi = orbit
			E = .5*(vr**2.+vtheta**2.+vphi**2.) -_G*self.M/(self.b + np.sqrt(r**2. + self.b**2.))
			L = r*np.sqrt(vtheta**2.+vphi**2.)
			Jphi = r*np.sin(theta)*vphi
			Jr = _G*self.M/np.sqrt(-2.*E) - .5*(L+np.sqrt(L**2.+4.*_G*self.M*self.b))
			return (Jphi,L-np.abs(Jphi),Jr) 

	def findJs(self,orbit,fixed_gauss=True):
		return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)