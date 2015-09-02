import numpy as np
from actions import *
from scipy.integrate import ode
import warnings

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

	def cartesian_force(self,x,y,z):
		"""Return -grad(phi) at the point x,y,z for orbit integration"""
		return -np.array([x,y,z/self.q**2.])*self.v0**2. / (x**2.+y**2.+(z/self.q)**2.)

	def circspeed(self,R):
		return np.sqrt(R * self.dR(R,0.))

	def findJs(self,orbit,delta=None,fixed_gauss=True):
		"""Find the actions for a given set of initial conditions"""
		if self.q!=1.:
			return get_actions_stackelfudge(orbit,self,delta=delta,fixed_gauss=fixed_gauss)
		else:
			return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)

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

	def cartesian_force(self,x,y,z):
		return -np.array([x,y,z/self.q**2.])*self.v0**2.*(x**2.+y**2.+(z/self.q)**2.)**(.5*self.gamma-1.)

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


	def cartesian_force(self,x,y,z):
		return -np.array([x,y,z*(self.a+np.sqrt(self.b**2.+z**2.))/np.sqrt(self.b**2.+z**2.)])*\
							_G*self.M*(x**2.+y**2.+(self.a+np.sqrt(self.b**2.+z**2.))**2.)**-1.5


	def findJs(self,orbit,delta=None,fixed_gauss=True):
		"""Find the actions for a given set of initial conditions"""
		if self.a==0.:
			return get_actions_spherical(orbit,self,fixed_gauss=fixed_gauss)
		else:
			return get_actions_stackelfudge(orbit,self,delta=delta,fixed_gauss=fixed_gauss)

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
		return -4.*np.pi*_G*self.rho0*self.rs**2.*np.log(1.+r/self.rs)*(self.rs/r)

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

	def cartesian_force(self,x,y,z):
		r = np.sqrt(x**2.+y**2.+z**2.)
		return np.array([x,y,z])*4.*np.pi*_G*self.rs**3.*self.rho0*((r**2.*(self.rs+r))**-1. -\
										 np.log(1.+r/self.rs)/r**3.)

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

	def cartesian_force(self,x,y,z):
		r = np.sqrt(x**2.+y**2.+z**2.)
		return -np.array([x,y,z])*_G*self.M/(r*(self.a+r)**2.)


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

	def cartesian_force(self,x,y,z):
		return np.sum(map(lambda f: f.cartesian_force(x,y,z), self.potlist),axis=0)

	def findJs(self,orbit,delta=None,fixed_gauss=True):
		return get_actions_stackelfudge(orbit,self,delta=delta,fixed_gauss=fixed_gauss)		


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

	def cartesian_force(self,x,y,z):
		r = np.sqrt(x**2.+y**2.+z**2.)
		rb = np.sqrt(self.b**2.+r**2.)
		return -np.array([x,y,z])*_G*self.M/(rb * (self.b + rb)**2.)


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



"""Integrate orbits, mostly so we can use Jason's generating function method"""

def gen_forces(t,x,pot):
    """gives the 6D derivatives for orbit integration, i.e. [vx,vy,vz,ax,ay,az]"""
    X,Y,Z=x[:3] #get the explicit coordinates
    return np.concatenate((x[3:],pot.cartesian_force(X,Y,Z)))

def integrate_orbit(x,tmax,pot):
	"""Integrate an orbit with initial coordinates x = [x,y,z,vx,vy,vz] for time tmax in potential pot"""
	solver = ode(gen_forces).set_integrator('dopri5',nsteps=1,rtol=1e-19,atol=1e-10)
	solver.set_initial_value(x,0.).set_f_params(pot)
	solver._integrator.iwork[2]=-1
	warnings.filterwarnings("ignore",category=UserWarning)
	t = np.array([0.])
	while solver.t < tmax:
		solver.integrate(tmax)
		x=np.vstack((x,solver.y))
		t=np.append(t,solver.t)
	warnings.resetwarnings()
	return x,t

def leapfrog_integrator(x,tmax,NT,Pot):
    deltat = tmax/NT
    h = deltat/100.
    t = 0.
    counter = 0
    X = np.copy(x)
    results = np.array([x])
    while(t<tmax):
        X[3:] += 0.5*h*Pot.cartesian_force(X[0],X[1],X[2])
        X[:3] += h*X[3:]
        X[3:] += 0.5*h*Pot.cartesian_force(X[0],X[1],X[2])
        # if(t==0.1):
        if(counter % 100 == 0):
            results=np.vstack((results,X))
        t+=h
        counter+=1
    return results,np.linspace(0.,tmax,NT)