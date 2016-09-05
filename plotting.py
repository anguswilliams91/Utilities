"""General purpose plotting tools"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
from matplotlib.colors import LogNorm
from matplotlib.pyplot import rc, axes
from sys import stderr
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import pylab
import matplotlib.colors as colors
import gus_utils as gu

#added a useless comment


def kde_smooth(x,y,ax=None,xlims=None,ylims=None,linecolor='k',ninterp=200,linewidth=2.,ncontours=10,fill=None,cmap="YlGnBu"):
    """Smooth density estimator of scattered data to make better contour plots n.b. this is 
    going to be slow so only use it if you have sparse data"""
    vals = np.vstack((x.ravel(),y.ravel())) #can pass arrays of any shape
    kernel = gaussian_kde(vals) #make the kde
    if xlims is None:
        xmin,xmax = np.min(x),np.max(x)
        ymin,ymax = np.min(y),np.max(y)
    else:
        xmin,xmax = xlims
        ymin,ymax = ylims
    X,Y = np.mgrid[xmin:xmax:complex(0,ninterp),ymin:ymax:complex(0,ninterp)] #grid of points for interpolation
    pos = np.vstack((X.ravel(),Y.ravel())) #shape required by kernel
    Z = np.reshape(kernel(pos).T,X.shape) #interpolated values
    if ax is None:
        if fill:
            plt.contourf(X,Y,Z,ncontours,cmap=cmap)
            plt.colorbar()
        plt.contour(X,Y,Z,ncontours,colors=linecolor,linewidths=linewidth)
    else:
        if fill:
            ax.contourf(X,Y,Z,ncontours,cmap=cmap)
            plt.colorbar()
        ax.contour(X,Y,Z,ncontours,colors=linecolor,linewidths=linewidth)
    return None        


def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '${:g}$'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str

def triangle_plot( chain, axis_labels=None, fname = None, nbins=40, filled=True, cmap="Blues", norm = None, truths = None,\
                         burnin=None, fontsize=20 , tickfontsize=15, nticks=4, linewidth=1.,wspace=0.5,hspace=0.5):

    """Plot a corner plot from an MCMC chain. the shape of the chain array should be (nwalkers*nsamples, ndim + 1). The extra column is for the walker ID 
    number (i.e. if you have 20 walkers the id numbers are np.arange(20)). Note the walker ID's are never used, theyre only assumed to be there because 
    of the way I write MCMC chains to file."""

    major_formatter = FuncFormatter(my_formatter)
    nwalkers = len(np.unique(chain[:,0]))

    if burnin is not None:
        traces = chain[nwalkers*burnin:,1:].T
    else:  
        traces = chain[:,1:].T

    if axis_labels is None:
        axis_labels = ['']*len(traces)

    #Defines the widths of the plots in inches
    plot_width = 15.
    plot_height = 15.
    axis_space = 0.05

    if len(traces) != len(axis_labels):
        print >> stderr, "ERROR: There must be the same number of axis labels as traces"
        return

    if truths != None and ( len(truths) != len(traces) ):
        print >> stderr, "ERROR: There must be the same number of true values as traces"

    num_samples = min([ len(trace) for trace in traces])
    n_traces = len(traces)


    #Set up the figure
    fig = plt.figure( num = None, figsize = (plot_height,plot_width))

    dim = 2*n_traces - 1

    gs = gridspec.GridSpec(dim+1,dim+1)
    gs.update(wspace=wspace,hspace=hspace)

    hist_2d_axes = {}

    #Create axes for 2D histograms
    for x_pos in xrange( n_traces - 1 ):
        for y_pos in range( n_traces - 1 - x_pos  ):
            x_var = x_pos
            y_var = n_traces - 1 - y_pos

            hist_2d_axes[(x_var, y_var)] = fig.add_subplot( \
                                           gs[ -1-(2*y_pos):-1-(2*y_pos), \
                                               2*x_pos:(2*x_pos+2) ] )
            hist_2d_axes[(x_var, y_var)].xaxis.set_major_formatter(major_formatter)
            hist_2d_axes[(x_var, y_var)].yaxis.set_major_formatter(major_formatter)

    #Create axes for 1D histograms
    hist_1d_axes = {}
    for var in xrange( n_traces -1 ):
        hist_1d_axes[var] = fig.add_subplot( gs[ (2*var):(2*var+2), 2*var:(2*var+2) ] )
        hist_1d_axes[var].xaxis.set_major_formatter(major_formatter)
        hist_1d_axes[var].yaxis.set_major_formatter(major_formatter)
    hist_1d_axes[n_traces-1] = fig.add_subplot( gs[-2:, -2:] )
    hist_1d_axes[n_traces-1].xaxis.set_major_formatter(major_formatter)
    hist_1d_axes[n_traces-1].yaxis.set_major_formatter(major_formatter)



    #Remove the ticks from the axes which don't need them
    for x_var in xrange( n_traces -1 ):
        for y_var in xrange( 1, n_traces - 1):
            try:
                hist_2d_axes[(x_var,y_var)].xaxis.set_visible(False)
            except KeyError:
                continue
    for var in xrange( n_traces - 1 ):
        hist_1d_axes[var].set_xticklabels([])
        hist_1d_axes[var].xaxis.set_major_locator(MaxNLocator(nticks))
        hist_1d_axes[var].yaxis.set_visible(False)

    for y_var in xrange( 1, n_traces ):
        for x_var in xrange( 1, n_traces - 1):
            try:
                hist_2d_axes[(x_var,y_var)].yaxis.set_visible(False)
            except KeyError:
                continue

    #Do the plotting
    #Firstly make the 1D histograms
    vals, walls = np.histogram(traces[-1][:num_samples], bins=nbins, normed = True)

    xplot = np.zeros( nbins*2 + 2 )
    yplot = np.zeros( nbins*2 + 2 )
    for i in xrange(1, nbins * 2 + 1 ):
        xplot[i] = walls[(i-1)/2]
        yplot[i] = vals[ (i-2)/2 ]

    xplot[0] = walls[0]
    xplot[-1] = walls[-1]
    yplot[0] = yplot[1]
    yplot[-1] = yplot[-2]

    Cmap = colors.Colormap(cmap)
    cNorm = colors.Normalize(vmin=0.,vmax=1.)
    scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)
    cVal = scalarMap.to_rgba(0.65)

    #this one's special, so do it on it's own
    hist_1d_axes[n_traces - 1].plot(xplot, yplot, color = 'k', lw=linewidth)
    if filled: hist_1d_axes[n_traces - 1].fill_between(xplot,yplot,color=cVal)
    hist_1d_axes[n_traces - 1].set_xlim( walls[0], walls[-1] )
    hist_1d_axes[n_traces - 1].set_xlabel(axis_labels[-1],fontsize=fontsize)
    hist_1d_axes[n_traces - 1].tick_params(labelsize=tickfontsize)
    hist_1d_axes[n_traces - 1].xaxis.set_major_locator(MaxNLocator(nticks))
    hist_1d_axes[n_traces - 1].yaxis.set_visible(False)
    plt.setp(hist_1d_axes[n_traces - 1].xaxis.get_majorticklabels(), rotation=45)
    if truths is not None:
        xlo,xhi = hist_1d_axes[n_traces-1].get_xlim()
        if truths[n_traces-1]<xlo:
            dx = xlo-truths[n_traces-1]
            hist_1d_axes[n_traces-1].set_xlim((xlo-dx-0.05*(xhi-xlo),xhi))
        elif truths[n_traces-1]>xhi:
            dx = truths[n_traces-1]-xhi
            hist_1d_axes[n_traces-1].set_xlim((xlo,xhi+dx+0.05*(xhi-xlo)))
        hist_1d_axes[n_traces - 1].axvline(truths[n_traces - 1],ls='--',c='k')


    #Now Make the 2D histograms
    for x_var in xrange( n_traces ):
        for y_var in xrange( n_traces):
            try:
                if norm == 'log':
                    H, y_edges, x_edges = np.histogram2d( traces[y_var][:num_samples], traces[x_var][:num_samples],\
                                                           bins = nbins, norm = LogNorm() )
                else:
                    H, y_edges, x_edges = np.histogram2d( traces[y_var][:num_samples], traces[x_var][:num_samples],\
                                                           bins = nbins )
                confidence_2d(traces[x_var][:num_samples],traces[y_var][:num_samples],ax=hist_2d_axes[(x_var,y_var)],\
                    nbins=nbins,intervals=None,linecolor='0.5',filled=filled,cmap=cmap,linewidth=linewidth)
                if truths is not None:
                    xlo,xhi = hist_2d_axes[(x_var,y_var)].get_xlim()
                    ylo,yhi = hist_2d_axes[(x_var,y_var)].get_ylim()
                    if truths[x_var]<xlo:
                        dx = xlo-truths[x_var]
                        hist_2d_axes[(x_var,y_var)].set_xlim((xlo-dx-0.05*(xhi-xlo),xhi))
                    elif truths[x_var]>xhi:
                        dx = truths[x_var]-xhi
                        hist_2d_axes[(x_var,y_var)].set_xlim((xlo,xhi+dx+0.05*(xhi-xlo)))
                    if truths[y_var]<ylo:
                        dy = ylo - truths[y_var]
                        hist_2d_axes[(x_var,y_var)].set_ylim((ylo-dy-0.05*(yhi-ylo),yhi))
                    elif truths[y_var]<ylo:
                        dy = truths[y_var] - yhi
                        hist_2d_axes[(x_var,y_var)].set_ylim((ylo,yhi+dy+0.05*(yhi-ylo)))
                    #TODO: deal with the pesky case of a prior edge
                    hist_2d_axes[(x_var,y_var)].plot( truths[x_var], truths[y_var], '*', color = 'k', markersize = 10 )
            except KeyError:
                pass
        if x_var < n_traces - 1:
            vals, walls = np.histogram( traces[x_var][:num_samples], bins=nbins, normed = True )

            xplot = np.zeros( nbins*2 + 2 )
            yplot = np.zeros( nbins*2 + 2 )
            for i in xrange(1, nbins * 2 + 1 ):
                xplot[i] = walls[(i-1)/2]
                yplot[i] = vals[ (i-2)/2 ]

            xplot[0] = walls[0]
            xplot[-1] = walls[-1]
            yplot[0] = yplot[1]
            yplot[-1] = yplot[-2]

            hist_1d_axes[x_var].plot(xplot, yplot, color = 'k' , lw=linewidth)
            if filled: hist_1d_axes[x_var].fill_between(xplot,yplot,color=cVal)
            hist_1d_axes[x_var].set_xlim( x_edges[0], x_edges[-1] )
            if truths is not None:
                xlo,xhi = hist_1d_axes[x_var].get_xlim()
                if truths[x_var]<xlo:
                    dx = xlo-truths[x_var]
                    hist_1d_axes[x_var].set_xlim((xlo-dx-0.05*(xhi-xlo),xhi))
                elif truths[x_var]>xhi:
                    dx = truths[x_var]-xhi
                    hist_1d_axes[x_var].set_xlim((xlo,xhi+dx+0.05*(xhi-xlo)))
                hist_1d_axes[x_var].axvline(truths[x_var],ls='--',c='k')

    #Finally Add the Axis Labels
    for x_var in xrange(n_traces - 1):
        hist_2d_axes[(x_var, n_traces-1)].set_xlabel(axis_labels[x_var],fontsize=fontsize)
        hist_2d_axes[(x_var, n_traces-1)].tick_params(labelsize=tickfontsize)
        hist_2d_axes[(x_var, n_traces-1)].xaxis.set_major_locator(MaxNLocator(nticks))
        plt.setp(hist_2d_axes[(x_var, n_traces-1)].xaxis.get_majorticklabels(), rotation=45)
    for y_var in xrange(1, n_traces ):
        hist_2d_axes[(0,y_var)].set_ylabel(axis_labels[y_var],fontsize=fontsize)
        hist_2d_axes[(0,y_var)].tick_params(labelsize=tickfontsize)
        plt.setp(hist_2d_axes[(0,y_var)].yaxis.get_majorticklabels(), rotation=45)
        hist_2d_axes[(0,y_var)].yaxis.set_major_locator(MaxNLocator(nticks))

    if fname != None:
        if len(fname.split('.')) == 1:
            fname += '.pdf'
        plt.savefig(fname, transparent=True, bbox_inches = "tight")

    return None

def PlotTraces(chain,burnin=None,axis_labels=None,nticks=4,tickfontsize=10,labelsize=20,truths=None):
    """Plot the traces of all the walkers in a given run"""
    c = gu.reshape_chain(chain)
    if burnin is not None:
        c=c[:,burnin:,:]
    nwalkers,nsteps,ndim = np.shape(c)
    if axis_labels is not None and len(axis_labels)!=ndim:
        print "You've messed up the number of axis labels"
        return None
    fig = plt.figure( num = None, figsize = (ndim,9.*ndim))
    gs = gridspec.GridSpec(ndim,1)
    gs.update(wspace=0.,hspace=0.)
    for i in np.arange(ndim):
        ax = fig.add_subplot(gs[i,0])
        ax.tick_params(labelsize=tickfontsize)
        ax.yaxis.set_major_locator(MaxNLocator(nticks))
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=45)
        if i!=ndim-1:
            ax.xaxis.set_visible(False)
            ticks = ax.yaxis.get_major_ticks()
            ticks[-1].set_visible(False)
            ticks[0].set_visible(False)
        if i==ndim-1:
            ticks = ax.yaxis.get_major_ticks()
            ticks[-1].set_visible(False)
            ax.set_xlabel("$N_\\mathrm{steps}$",fontsize=labelsize)
        [ax.plot(c[j,:,i],c='0.5',alpha=0.3) for j in np.arange(nwalkers)]
        if truths is not None:
            ax.axhline(truths[i],ls='--',c='k')
        ax.set_xlim((0,nsteps))
        if axis_labels is not None:
            ax.set_ylabel(axis_labels[i],fontsize=labelsize)
    return None


def gus_contour(x,y,nbins=20,ncontours=10,log=False,histunder=False,cmap="YlGnBu",linecolor='k',ax=None,interp='nearest',tickfontsize=15):
    """Make a basic contour plot from scattered data, interp can be 'nearest','bilinear','bicubic' etc."""
    H,xedges,yedges = np.histogram2d(y,x,bins=nbins)
    extent = [yedges[0],yedges[-1],xedges[0],xedges[-1]]
    if ax is None:
        if not log: plt.contour(H,ncontours,extent=extent,colors=linecolor)
        else:
            levels = np.logspace(.2*np.max(np.log(H[H!=0.])),np.max(np.log(H[H!=0.])),ncontours)
            plt.contour(H,extent=extent,colors=linecolor,norm=LogNorm(),levels=levels)
        if histunder and not log:
            plt.imshow(H,interpolation=interp,extent=extent,origin='lower',cmap=cmap)
        elif histunder:
            plt.imshow(H,interpolation=interp,extent=extent,origin='lower',norm=LogNorm(),cmap=cmap)
        plt.gca().set_aspect("auto")
        plt.gca().tick_params(labelsize=tickfontsize)
        plt.gca().set_aspect("auto")
    else:
        if not log: ax.contour(H,ncontours,extent=extent,colors=linecolor)
        else:
            levels = np.logspace(.2*np.max(np.log10(H[H!=0.])),np.max(np.log(H[H!=0.])),ncontours)
            ax.contour(H,extent=extent,colors=linecolor,norm=LogNorm(),levels=levels)
        if histunder and not log:
            ax.imshow(H,interpolation=interp,extent=extent,origin='lower',cmap=cmap)
        elif histunder:
            ax.imshow(H,interpolation=interp,extent=extent,origin='lower',norm=LogNorm(),cmap=cmap)  
        ax.set_aspect("auto")
        ax.tick_params(labelsize=tickfontsize)
        ax.set_aspect("auto")
    return None

def scalarmap(x,y,s,nbins=10,ncontours=10,logdens=False,logscalar=False,cmap="YlGnBu",linecolor='k',ax=None,interp='nearest',dispersion=False):
    """Plot a map of the scalar function s as a function of x and y, given irregular data. Overplot contours of x and y. The mean of 
    s in each bin is plotted"""
    if logscalar is True and any(s<0.) is True:
        print "Can't log scale a quantity that isn't positive definite!"
        return None 
    H,yedges,xedges = np.histogram2d(y,x,bins=nbins) #histogram the data
    if not dispersion:
        H_s,yedges,xedges = np.histogram2d(y,x,weights=s,bins=nbins) #histogram with the scalar as the weight so that H_s/H is the mean of s in each bin
        H_s/=H
    else:
        H_m,yedges,xedges = np.histogram2d(y,x,weights=s,bins=nbins) #histogram with the scalar as the weight so that H_s/H is the mean of s in each bin
        H_m2,yedges,xedges = np.histogram2d(y,x,weights=s*s,bins=nbins) #histogram with the scalar as the weight so that H_s/H is the mean of the square s in each bin
        H_s = np.sqrt(H_m2/H - (H_m/H)**2.) #the dispersion in each pixel
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    if ax is None:
        if not logdens: plt.contour(H,ncontours,extent=extent,colors=linecolor)
        else:
            levels = np.logspace(.2*np.max(np.log(H[H!=0.])),np.max(np.log(H[H!=0.])),ncontours)
            plt.contour(H,extent=extent,colors=linecolor,norm=LogNorm(),levels=levels)
        if not logscalar:
            plt.imshow(H_s,interpolation=interp,extent=extent,origin='lower',cmap=cmap)
        else:
            plt.imshow(H_s,interpolation=interp,extent=extent,origin='lower',norm=LogNorm(),cmap=cmap)
        plt.gca().set_aspect("auto")
    else:
        if not logdens: ax.contour(H,ncontours,extent=extent,colors=linecolor)
        else:
            levels = np.logspace(.2*np.max(np.log10(H[H!=0.])),np.max(np.log(H[H!=0.])),ncontours)
            ax.contour(H,extent=extent,colors=linecolor,norm=LogNorm(),levels=levels)
        if not logscalar:
            ax.imshow(H_s,interpolation=interp,extent=extent,origin='lower',cmap=cmap)
        else:
            ax.imshow(H_s,interpolation=interp,extent=extent,origin='lower',norm=LogNorm(),cmap=cmap)  
        ax.set_aspect("auto")
    return xedges,yedges,H,H_s 

def scalarmap1D(x,s=None,nbins=10,ax=None,log=False,errors=True,linecolor='k'):
    """same as above but for 1D case, if s is None then just does a density plot"""
    H,xedges = np.histogram(x,bins=nbins)
    if s is not None:
        H_s,xedges = np.histogram(x,weights=s,bins=nbins)
        fs = H_s/H
        if errors:
            disp = np.array([np.std(s[(x<xedges[i+1])*(x>=xedges[i])])/np.sqrt(np.float(H[i])) for i in np.arange(nbins-1)])
            i = nbins -1
            disp = np.append(disp,np.std(s[(x<=xedges[i+1])*(x>=xedges[i])])/np.sqrt(np.float(H[i])))
            print disp
    else:
        fs = H
    xc = np.array([np.mean([xedges[i],xedges[i+1]]) for i in np.arange(nbins)])
    if ax is None and log:
        plt.loglog(xc,fs,c=linecolor)
    elif ax is None and not log and not errors:
        plt.plot(xc,fs,c=linecolor)
    elif ax is not None and log:
        ax.loglog(xc,fs,c=linecolor)
    elif errors and ax is None:
        plt.errorbar(xc,fs,yerr=disp,c=linecolor)
    elif errors and ax is not None:
        ax.errorbar(xc,fs,yerr=disp,c=linecolor)
    else:
        ax.plot(xc,fs,c=linecolor)
    return None

def vectormap(x,y,vx,vy,nbins=10,ax=None,cmap="YlGnBu",colorlines=False,density=1.,linecolor='k'):
    """Make a streamplot of a 2D vector quantity, but averaged in a grid of pixels"""
    #Simon's hack for getting the means quick
    H,yedges,xedges = np.histogram2d(y,x,bins=nbins) #density histogram 
    Hx,yedges,xedges = np.histogram2d(y,x,bins=nbins,weights=vx) #sum of x-component of vector in each pixel
    Hy,yedges,xedges = np.histogram2d(y,x,bins=nbins,weights=vy) #sum of y-component of vector in each pixel
    vxm,vym = Hx/H,Hy/H #the means
    #bin centres
    xc = np.array([.5*(xedges[i]+xedges[i+1]) for i in np.arange(nbins)])
    yc = np.array([.5*(yedges[i]+yedges[i+1]) for i in np.arange(nbins)])
    #meshgrid to make arrays for streamplot
    xx,yy = np.meshgrid(xc,yc)
    if ax is None and colorlines is False:
        plt.streamplot(xx,yy,vxm,vym,density=density,color=linecolor)
        plt.xlim((xedges[0],xedges[-1]))
        plt.ylim((yedges[0],yedges[-1]))
    elif ax is None and colorlines is True:
        plt.streamplot(xx,yy,vxm,vym,density=density,color=np.sqrt(vxm**2.+vym**2.),cmap=cmap)
        plt.xlim((xedges[0],xedges[-1]))
        plt.ylim((yedges[0],yedges[-1]))
    elif ax is not None and colorlines is False:
        ax.streamplot(xx,yy,vxm,vym,density=density,color=linecolor)
        ax.set_xlim((xedges[0],xedges[-1]))
        ax.set_ylim((yedges[0],yedges[-1]))
    else:
        ax.streamplot(xx,yy,vxm,vym,density=density,color=np.sqrt(vxm**2.+vym**2.),cmap=cmap)
        ax.set_xlim((xedges[0],xedges[-1]))
        ax.set_ylim((yedges[0],yedges[-1]))
    return None


def confidence_2d(xsamples,ysamples,ax=None,intervals=None,nbins=20,linecolor='k',histunder=False,cmap="Blues",filled=False,linewidth=1.):
    """Draw confidence intervals at the levels asked from a 2d sample of points (e.g. 
        output of MCMC)"""
    if intervals is None:
        intervals  = 1.0 - np.exp(-0.5 * np.array([0., 1., 2.]) ** 2)
    H,yedges,xedges = np.histogram2d(ysamples,xsamples,bins=nbins)


    #get the contour levels
    h = H.flatten()
    h = h[np.argsort(h)[::-1]]
    sm = np.cumsum(h)
    sm/=sm[-1]
    v = np.empty(len(intervals))
    for i,v0 in enumerate(intervals):
        try:
            v[i] = h[sm <= v0][-1]
        except:
            v[i] = h[0]
    v =v[::-1]

    xc = np.array([.5*(xedges[i]+xedges[i+1]) for i in np.arange(nbins)]) #bin centres
    yc = np.array([.5*(yedges[i]+yedges[i+1]) for i in np.arange(nbins)])

    xx,yy = np.meshgrid(xc,yc)

    if ax is None:
        if histunder:
            plt.hist2d(xsamples,ysamples,bins=nbins,cmap=cmap)
            plt.contour(xx,yy,H,levels=v,colors=linecolor,extend='max',linewidths=linewidth)
        elif filled:
            plt.contourf(xx,yy,H,levels=v,cmap=cmap)
        else:
            plt.contour(xx,yy,H,levels=v,colors=linecolor,linewidths=linewidth)
    else:
        if histunder:
            ax.hist2d(xsamples,ysamples,bins=nbins,cmap=cmap)
            ax.contour(xx,yy,H,levels=v,colors=linecolor,extend='max',linewidths=linewidth)
        elif filled:
            ax.contourf(xx,yy,H,levels=v,cmap=cmap)
            ax.contour(xx,yy,H,levels=v,colors=linecolor,extend='max',linewidths=linewidth)
        else:
            ax.contour(xx,yy,H,levels=v,colors=linecolor,linewidths=linewidth)        

    return None

def posterior_1D(paramsamples,x,func,burnin=None,axis_labels=None,ax=None,cmap="Blues",alpha=1.,fill=True,fontsize=20,tickfontsize=20):
    """Given an MCMC output paramsamples.shape = (nparams,N) a 1D function func(x,params) that 
        depends on params and a given range x, use the samples to create a plot with confidence 
        intervals on the derived parameters. func should be vectorized."""
    cm = plt.cm.get_cmap(cmap) #get the cmap
    if burnin is not None:
        paramsamples = paramsamples[:,burnin:]
    nparams,N = np.shape(paramsamples)
    funsamples = np.zeros((len(x),N))
    #compute the MCMC samples of the function at each position
    for i in np.arange(len(x)):
        funsamples[i] = func(x[i],paramsamples)
    #now compute the confidence intervals of the function at each position
    confs = np.zeros((len(x),5))
    for i in np.arange(len(x)):
        confs[i,0] = np.percentile(funsamples[i],3)
        confs[i,1] = np.percentile(funsamples[i],16)
        confs[i,2] = np.percentile(funsamples[i],50)
        confs[i,3] = np.percentile(funsamples[i],84)
        confs[i,4] = np.percentile(funsamples[i],97)
    #now plot 
    if ax is not None and fill is True:
        ax.plot(x,confs[:,2],c=cm(1.))
        ax.fill_between(x,confs[:,1],confs[:,3],facecolor=cm(0.25),lw=0,alpha=alpha)
        ax.fill_between(x,confs[:,3],confs[:,4],facecolor=cm(0.75),lw=0,alpha=alpha)
        ax.fill_between(x,confs[:,0],confs[:,1],facecolor=cm(0.75),lw=0,alpha=alpha)
        ax.set_xlim((np.min(x),np.max(x)))
        ax.tick_params(labelsize=tickfontsize)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
    elif ax is None and fill is True:
        plt.plot(x,confs[:,2],c=cm(1.))
        plt.fill_between(x,confs[:,1],confs[:,3],facecolor=cm(0.7),lw=0,alpha=alpha)
        plt.fill_between(x,confs[:,3],confs[:,4],facecolor=cm(0.3),lw=0,alpha=alpha)
        plt.fill_between(x,confs[:,0],confs[:,1],facecolor=cm(0.3),lw=0,alpha=alpha)
        plt.xlim((np.min(x),np.max(x)))
        plt.gca().tick_params(labelsize=tickfontsize)
        if axis_labels is not None:
            plt.xlabel(axis_labels[0],fontsize=fontsize)
            plt.ylabel(axis_labels[1],fontsize=fontsize)
    elif ax is not None and fill is False:
        #only plot the 1 sigma lines if no fill, otherwise it looks too messy
        ax.plot(x,confs[:,1],c=cm(.5))
        ax.plot(x,confs[:,2],c=cm(.5))
        ax.plot(x,confs[:,3],c=cm(.5))
        ax.set_xlim((np.min(x),np.max(x)))
        ax.tick_params(labelsize=tickfontsize)
        if axis_labels is not None:
            plt.xlabel(axis_labels[0],fontsize=fontsize)
            plt.ylabel(axis_labels[1],fontsize=fontsize)
    else:
        plt.plot(x,confs[:,1],c=cm(.5))
        plt.plot(x,confs[:,2],c=cm(.5))
        plt.plot(x,confs[:,3],c=cm(.5))   
        plt.xlim((np.min(x),np.max(x)))
        plt.gca().tick_params(labelsize=tickfontsize)
        if axis_labels is not None:
            plt.xlabel(axis_labels[0],fontsize=fontsize)
            plt.ylabel(axis_labels[1],fontsize=fontsize)      
    return None

def regular_contour(x,y,z,ncontours=20,linewidth=0.5,linecolor='k',cmap="YlGnBu",labels=None,ax=None,fname=None,aspect_ratio="auto",\
                        cbar_orientation='vertical'):
    """Plot a contour map which consists of filled contours and line contours"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    fc = ax.contourf(x,y,z,ncontours,cmap=cmap) #do the filled contours
    if labels is not None: #add labels if provided
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        plt.colorbar(fc,label=labels[2],orientation=cbar_orientation)
    else: #if not, just do the colorbar
        plt.colorbar(fc,orientation=cbar_orientation)
    if linewidth>0.: ax.contour(x,y,z,ncontours,linewidths=linewidth,colors=linecolor) #now do the lines
    ax.set_aspect(aspect_ratio)
    if fname is not None:
        fig.savefig(fname) #save if path provided
    return None
