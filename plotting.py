"""General purpose plotting tools"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import kde
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
from matplotlib.colors import LogNorm
from matplotlib.pyplot import rc, axes
from sys import stderr
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import pylab
import matplotlib.colors as colors


def scatter_contour(x, y,
                levels=10,
                    threshold=100,
                            log_counts=False,
                                        ax=None,
                                            kdebins = 20):

    """Contour plots with underlying 2d histograms and scatter points below a certain threshold"""

    x = np.asarray(x)
    y = np.asarray(y)
    default_contour_args = dict(zorder=0)
    default_plot_args = dict(marker='.', linestyle='none', zorder=1)

    if ax is None:
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y)
    
    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)
    

    X = np.hstack([x[:, None], y[:, None]])

    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                                    linewidths=0, extent=extent,
                                                        alpha=0)

    if len(outline.allsegs[0]) > 0:
        outer_poly = outline.allsegs[0][0]
        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]
        Yplot = X[points_inside]
    else:
        Xplot = X


    xkde = np.ravel(Yplot[:,0])
    ykde = np.ravel(Yplot[:,1])
    k = kde.gaussian_kde(np.vstack((xkde,ykde)))
    xi,yi = np.mgrid[xkde.min():xkde.max():kdebins*1j, ykde.min():ykde.max():kdebins*1j]
    zi = k(np.vstack([xi.flatten(),yi.flatten()]))
    contours = ax.contour(xi,yi,zi.reshape(xi.shape),extent=extent,colors='0.3')
    ax.hist2d(x,y,alpha=0.6,bins=10,cmin=threshold,cmap = 'hot_r' )

    points = ax.scatter(Xplot[:, 0], Xplot[:, 1],c='k',alpha=0.2,edgecolors='none')
    plt.xlim((np.min(Xplot[:,0]),np.max(Xplot[:,0])))
    plt.ylim((np.min(Xplot[:,1]),np.max(Xplot[:,1])))
    plt.tight_layout

    return points, contours

def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str

def triangle_plot( chain, axis_labels=None, fname = None, nbins=40, filled=True, cmap="Greens", norm = None, truevals = None, burnin=None, fontsize=20 , tickfontsize=15, nticks=4):

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

    if truevals != None and ( len(truevals) != len(traces) ):
        print >> stderr, "ERROR: There must be the same number of true values as traces"

    num_samples = min([ len(trace) for trace in traces])
    n_traces = len(traces)


    #Set up the figure
    fig = plt.figure( num = None, figsize = (plot_height,plot_width))

    dim = 2*n_traces - 1

    gs = gridspec.GridSpec(dim+1,dim+1)
    gs.update(wspace=0.5,hspace=0.5)

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

    hist_1d_axes[n_traces - 1].plot(xplot, yplot, color = 'k')
    hist_1d_axes[n_traces - 1].fill_between(xplot,yplot,color=cVal)
    hist_1d_axes[n_traces - 1].set_xlim( walls[0], walls[-1] )
    hist_1d_axes[n_traces - 1].set_xlabel(axis_labels[-1],fontsize=fontsize)
    hist_1d_axes[n_traces - 1].tick_params(labelsize=tickfontsize)
    hist_1d_axes[n_traces - 1].xaxis.set_major_locator(MaxNLocator(nticks))
    hist_1d_axes[n_traces - 1].yaxis.set_visible(False)
    plt.setp(hist_1d_axes[n_traces - 1].xaxis.get_majorticklabels(), rotation=45)


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
                confidence_2d(traces[x_var][:num_samples],traces[y_var][:num_samples],ax=hist_2d_axes[(x_var,y_var)],nbins=nbins,intervals=None,linecolor='0.5',filled=filled,cmap=cmap)
                if truevals != None:
                    hist_2d_axes[(x_var,y_var)].plot( truevals[x_var], truevals[y_var], '+', color = '0.3', markersize = 30 )
                    hist_2d_axes[(x_var, y_var)].set_xlim( extent[0], extent[1] )
                    hist_2d_axes[(x_var, y_var)].set_ylim( extent[2], extent[3] )
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

            hist_1d_axes[x_var].plot(xplot, yplot, color = 'k' )
            hist_1d_axes[x_var].fill_between(xplot,yplot,color=cVal)

            hist_1d_axes[x_var].set_xlim( x_edges[0], x_edges[-1] )

    #Finally Add the Axis Labels
    for x_var in xrange(n_traces - 1):
        hist_2d_axes[(x_var, n_traces-1)].set_xlabel(axis_labels[x_var],fontsize=fontsize)
        hist_2d_axes[(x_var, n_traces-1)].tick_params(labelsize=tickfontsize)
        hist_2d_axes[(x_var, n_traces-1)].xaxis.set_major_locator(MaxNLocator(nticks))
        plt.setp(hist_2d_axes[(x_var, n_traces-1)].xaxis.get_majorticklabels(), rotation=45)
    for y_var in xrange(1, n_traces ):
        hist_2d_axes[(0,y_var)].set_ylabel(axis_labels[y_var],fontsize=fontsize)
        hist_2d_axes[(0,y_var)].tick_params(labelsize=tickfontsize)
        hist_2d_axes[(0,y_var)].yaxis.set_major_locator(MaxNLocator(nticks))

    if fname != None:
        if len(fname.split('.')) == 1:
            fname += '.eps'
        plt.savefig(fname, transparent=True, bbox_inches = "tight")

    return None


def gus_contour(x,y,nbins=20,ncontours=10,log=False,histunder=False,cmap="hot_r",linecolor='k',ax=None,interp='nearest',tickfontsize=15):
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

def scalarmap(x,y,s,nbins=10,ncontours=10,logdens=False,logscalar=False,cmap="hot_r",linecolor='k',ax=None,interp='nearest',f = lambda x: np.mean(x)):
    """Plot a map of the scalar function s as a function of x and y, given irregular data. Overplot contours of x and y. The mean of 
    s in each bin is plotted"""
    if logscalar is True and any(s<0.) is True:
        print "Can't log scale a quantity that isn't positive definite!"
        return None 
    H,yedges,xedges = np.histogram2d(y,x,bins=nbins) #histogram the data
    H_s,yedges,xedges = np.histogram2d(y,x,weights=s,bins=nbins) #histogram with the scalar as the weight so that H_s/H is the mean of s in each bin
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    if ax is None:
        if not logdens: plt.contour(H,ncontours,extent=extent,colors=linecolor)
        else:
            levels = np.logspace(.2*np.max(np.log(H[H!=0.])),np.max(np.log(H[H!=0.])),ncontours)
            plt.contour(H,extent=extent,colors=linecolor,norm=LogNorm(),levels=levels)
        if not logscalar:
            plt.imshow(H_s/H,interpolation=interp,extent=extent,origin='lower',cmap=cmap)
        else:
            plt.imshow(H_s/H,interpolation=interp,extent=extent,origin='lower',norm=LogNorm(),cmap=cmap)
    else:
        if not logdens: ax.contour(H,ncontours,extent=extent,colors=linecolor)
        else:
            levels = np.logspace(.2*np.max(np.log10(H[H!=0.])),np.max(np.log(H[H!=0.])),ncontours)
            ax.contour(H,extent=extent,colors=linecolor,norm=LogNorm(),levels=levels)
        if not logscalar:
            ax.imshow(H_s/H,interpolation=interp,extent=extent,origin='lower',cmap=cmap)
        else:
            ax.imshow(H_s/H,interpolation=interp,extent=extent,origin='lower',norm=LogNorm(),cmap=cmap)  
    return None  

def scalarmap1D(x,s=None,nbins=10,ax=None,log=False):
    """same as above but for 1D case, if s is None then just does a density plot"""
    H,xedges = np.histogram(x,bins=nbins)
    if s is not None:
        H_s,xedges = np.histogram(x,weight=s,bins=nbins)
        fs = H_s/H
    else:
        fs = H
    xc = np.array([np.mean([xedges[i],xedges[i+1]]) for i in np.arange(nbins)])
    if ax is None and log:
        plt.loglog(xc,fs)
    elif ax is None and not log:
        plt.plot(xc,fs)
    elif ax is not None and log:
        ax.loglog(xc,fs)
    else:
        ax.plot(xc,fs)
    return None

def vectormap(x,y,vx,vy,nbins=10,ax=None,cmap="hot_r",colorlines=False,density=1.,linecolor='k'):
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


def confidence_2d(xsamples,ysamples,ax=None,intervals=None,nbins=20,linecolor='k',histunder=False,cmap="hot_r",filled=False):
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

    xc = np.array([.5*(xedges[i]+xedges[i+1]) for i in np.arange(nbins)]) #bin centres
    yc = np.array([.5*(yedges[i]+yedges[i+1]) for i in np.arange(nbins)])

    xx,yy = np.meshgrid(xc,yc)

    if ax is None:
        if histunder:
            plt.hist2d(xsamples,ysamples,bins=nbins,cmap=cmap)
            plt.contour(xx,yy,H,levels=v,colors=linecolor,extend='max')
        elif filled:
            plt.contourf(xx,yy,H,levels=v[::-1],cmap=cmap)
        else:
            plt.contour(xx,yy,H,levels=v,colors=linecolor)
    else:
        if histunder:
            ax.hist2d(xsamples,ysamples,bins=nbins,cmap=cmap)
            ax.contour(xx,yy,H,levels=v,colors=linecolor,extend='max')
        elif filled:
            ax.contourf(xx,yy,H,levels=v[::-1],cmap=cmap)
            ax.contour(xx,yy,H,levels=v,colors=linecolor,extend='max')
        else:
            ax.contour(xx,yy,H,levels=v,colors=linecolor)        

    return None

