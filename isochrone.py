import numpy as np 
from scipy.optimize import minimize_scalar
import gus_utils as gu


"""Code for evaluating various things in the Isochrone model like (J, theta) -> (x, v) and vice-versa.
    Note - Jphi can be negative here which isn't strictly correct since Jphi = |Lz| but is referred to 
    as so to avoid notational clutter"""

_G = 4299422.9171399993 #constant so that mass is 10^12 Msolar, velocities are kms^-1 and distances are kpc

class IsochroneModel():

    def __init__(self, M=2.7, b=8.):
        #take the model used by Sanderson et al. (2015)
        self._GM = _G*M
        self._b = b

    def __call__(self,r):
        #evaluate the potential
        return -self._GM * (self._b + np.sqrt(self._b**2. + r**2.))**-1.

    def H(self,J):
        #evaluate the Hamiltonian given actions
        Jr,Jtheta,Jphi = J
        L = Jtheta+np.abs(Jphi)
        return -2.*self._GM**2.*(2.*Jr+ L + np.sqrt(4.*self._b*self._GM+L**2.))**-2.

    def _geteta(self,eps,Tr):
        #solve for the variable eta from McGill & Binney (1990)
        etafun = lambda eta: np.abs(eta - eps*np.sin(eta) - Tr)
        try:
            res = minimize_scalar(etafun)
            return res.x
        except:
            print "Couldn't solve for eta: eps = {0}, Tr = {1}".format(eps,Tr)
            return None

    def _lambfun(self,eta,aa):
        #to ensure continuity of the angles
        if (0.<eta)*(np.pi/2.>eta):
            return np.arctan2(aa*np.sin(eta),1.+np.cos(eta))
        elif (np.pi/2.<eta)*(np.pi>eta):
            return np.pi/2. - np.arctan2(1.+np.cos(eta),np.sin(eta)*aa)
        elif (np.pi<eta)*(1.5*np.pi>eta):
            temp = np.arctan2(1.+np.cos(eta),np.sin(eta)*aa)
            if temp>0.:
                return 1.5*np.pi - temp
            else:
                return .5*np.pi - temp
        elif (1.5*np.pi<eta)*(eta<np.pi*2.):
            return np.pi + np.arctan2(aa*np.sin(eta),1.+np.cos(eta))
        else:
            print "eta = {} is outside the allowed range.".format(eta)
            return None


    def AA2xv(self,JAng,cartesian=False):
        #go from angle-actions to configuration space (spherical coordinates) following the steps from McGill & Binney (1990)

        Jr,Jt,Jp,Tr,Tt,Tp = JAng #unpack the actions and the angles
        Omega = Tp - np.sign(Jp)*Tt #longitude of ascending node
        L = Jt + np.abs(Jp)
        J1,J2,J3,T1,T2,T3 = Jp,L,Jr,Omega,Tt,Tr #match notation in B&T

        #compute various things we will need
        E = self.H([Jr,Jt,Jp])
        c = self._GM/(-2.*E) - self._b
        e = np.sqrt(1.-(J2**2./(self._GM*c))*(1.+self._b/c))
        eps = e*c/(c+self._b)
        eta = self._geteta(eps,T3) #solve the equation for eta numerically 

        #now get r and vr
        f = (c/self._b)*(1.-e*np.cos(eta))
        r = self._b*np.sqrt(f*(2.+f))
        Om3 = 8.*(self._GM)**2. / (2.*J3 + J2 + np.sqrt(J2**2. + 4.*self._GM*self._b))
        vr = np.sqrt(self._GM/(self._b+c))*(c*e*np.sin(eta))/r

        #get the phase in the orbital plane
        Omratio = .5*(1.+J2/np.sqrt(J2**2.+self._GM*4.*self._b))
        a1,a2 = np.sqrt((1.+e)/(1.-e)), np.sqrt((1.+e+2.*self._b/c)/(1.-e+2.*self._b/c))
        psi = T2 - Omratio*T3 + self._lambfun(eta,a1) + self._lambfun(eta,a2)/np.sqrt(1.+4.*self._GM*self._b/J2**2.)

        #get theta and vtheta
        i = np.arccos(J1/J2)
        theta = np.arccos(np.sin(i)*np.sin(psi))
        vtheta = -J2*np.sin(i)*np.cos(psi)/np.sin(theta)/r

        #get vphi
        vphi = J1 / (r*np.sin(theta))

        #get phi 
        sinu = np.sin(psi)*np.cos(i)/np.sin(theta)
        if sinu>1.:
            u = np.pi/2.
        if sinu<1.:
            u = -np.pi/2.
        else:
            u = np.arcsin(sinu)
        if vtheta>0.:
            u = np.pi-u

        phi = (u + T1) % (2.*np.pi)

        if cartesian:
            return gu.spherical2cartesian(r,theta,phi,vr,vtheta,vphi)
        else:
            return np.array([r,theta,phi,vr,vtheta,vphi])

    def xv2AA(self,xv,cartesian=False,JustActions=False):
        #go from positions and velocities to actions and angles
        if cartesian:
            x,y,z,vx,vy,vz = xv
            r,theta,phi,vr,vtheta,vphi = gu.cartesian2spherical(x,y,z,vx,vy,vz)
        else:
            r,theta,phi,vr,vtheta,vphi = xv

        E = .5*(vr*vr+vtheta*vtheta+vphi*vphi)+self.__call__(r)
        Jp = vphi*r*np.sin(theta)
        L = np.sqrt(vtheta**2.+vphi**2.)*r
        Jt = L - np.abs(Jp)
        Jr = self._GM/np.sqrt(-2.*E) - .5*(L+np.sqrt(L*L + 4.*self._GM*self._b))

        if not JustActions:
            #now get the angles
            c = self._GM/(-2.*E) - self._b
            e = np.sqrt(1. - (1.+self._b/c)*L*L/(self._GM*c))
            s = 1.+ np.sqrt(1.+r*r/self._b**2.)

            num = r*vr / np.sqrt(-2.*E)
            denom = self._b + c - np.sqrt(self._b*self._b + r*r)
            eta = np.arctan2(num,denom)
            eta %= (2.*np.pi)
            Tr = eta - e*c*np.sin(eta) / (c + self._b)


            psi = np.arctan2(np.cos(theta), -np.sin(theta)*r*vtheta/L)
            if np.abs(vtheta)<1e-10: psi = np.pi/2.

            Omratio = .5*(1.+L/np.sqrt(L**2.+self._GM*4.*self._b))
            a1,a2 = np.sqrt((1.+e)/(1.-e)), np.sqrt((1.+e+2.*self._b/c)/(1.-e+2.*self._b/c))
            Tt = psi + Omratio*Tr - self._lambfun(eta,a1) - self._lambfun(eta,a2)/np.sqrt(1.+4.*self._GM*self._b/L**2.)

            inc = Jp/L
            sinu = (inc/np.sqrt(1.-inc*inc)/np.tan(theta))
            if sinu>1.:
                u = np.pi/2.
            if sinu<1.:
                u = -np.pi/2.
            else:
                u = np.arcsin(sinu)
            if vtheta>0.:
                u = np.pi-u

            Tp = phi - u + np.sign(Jp)*Tt
            angs = np.array([Tr,Tt,Tp])
            angs %= (2.*np.pi)

            return np.append(np.array([Jr,Jt,Jp]),angs)
        else:
            return np.array([Jr,Jt,Jp])
