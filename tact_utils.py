"""Convenient python functions for using things in tact"""

import aa_py
import multiprocessing
import numpy as np

"""note that the order of actions/frequencies/angles is R,phi,z"""

def PifflPot():
	#quick function to return the Piffl (2014) galactic potential
	return aa_py.GalPot('/home/aamw3/Dropbox/PhD_dd/misc/DONUT/Torus/pot/Piffl14.Tpot')

def GetActions(w,pot,actionfinder):
	#some extras around Jason's action finding code - for now just check if the orbit is unbound
	x,y,z,vx,vy,vz = w
	e = .5*(vx**2.+vy**2.+vz**2.)+pot(np.array([x,y,z]))
	#some potentials don't tend to 0. as we approach spatial infinity
	if e>pot(1e3*np.ones(3)):
		raise ValueError
	else:
		return actionfinder.actions(np.array(w))

def GetAnglesFreqs(w,pot,actionfinder):
	#some extras around Jason's action finding code - for now just check if the orbit is unbound
	x,y,z,vx,vy,vz = w
	e = .5*(vx**2.+vy**2.+vz**2.)+pot(np.array([x,y,z]))
	#some potentials don't tend to 0. as we approach spatial infinity
	if e>pot(1e3*np.ones(3)):
		raise ValueError
	else:
		return actionfinder.angles(np.array(w))

def ManyActions(ws,pot,actionfinder):
	"""Calculate the actions for many points,
	shape of ws should be (N,6) where N is the number of points to calculate the 
	actions for. TODO: add multiprocessing"""
	ws = np.array(ws)
	return np.array([GetActions(wsi,pot,actionfinder) for wsi in ws])

def ManyAngFreqs(ws,pot,actionfinder):
	"""Calculate the actions for many points,
	shape of ws should be (N,6) where N is the number of points to calculate the 
	actions for. TODO: add multiprocessing"""
	return np.array([GetAnglesFreqs(wsi,pot,actionfinder) for wsi in ws])


