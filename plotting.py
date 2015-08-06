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

def triangle_plot( chain, axis_labels, fname = None, nbins=100, norm = None, truevals = None, display = False, burnin=None ):

    """Plot a corner plot from an MCMC chain"""

    major_formatter = FuncFormatter(my_formatter)

    if burnin is not None:
        traces = chain[burnin:,1:].T
    else:  
        traces = chain[:,1:].T

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
        hist_1d_axes[var].xaxis.set_visible(False)
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

    hist_1d_axes[n_traces - 1].plot(xplot, yplot, color = 'k')
    hist_1d_axes[n_traces - 1].fill_between(xplot,yplot,color='r',alpha=0.3)
    hist_1d_axes[n_traces - 1].set_xlim( walls[0], walls[-1] )
    hist_1d_axes[n_traces - 1].set_xlabel(axis_labels[-1])
    hist_1d_axes[n_traces - 1].xaxis.set_major_locator(MaxNLocator(4))
    hist_1d_axes[n_traces - 1].yaxis.set_visible(False)


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
                x_bin_sizes,y_bin_sizes = (x_edges[1:]-x_edges[:-1]).reshape((1,nbins)), (y_edges[1:]-y_edges[:-1])    
                X,Y = 0.5*(x_edges[1:]+x_edges[:-1]), 0.5*(y_edges[1:]+y_edges[:-1])
                pdf = (H*(x_bin_sizes*y_bin_sizes))
                H = H[::-1]
                extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
                hist_2d_axes[(x_var,y_var)].imshow(H, extent=extent, \
                             aspect='auto', interpolation='nearest',cmap='hot_r')
                hist_2d_axes[(x_var,y_var)].contour(X,Y,pdf,3,colors='0.5',linewidth=0.25)
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
            hist_1d_axes[x_var].fill_between(xplot,yplot,color='r',alpha=0.3)

            hist_1d_axes[x_var].set_xlim( x_edges[0], x_edges[-1] )

    #Finally Add the Axis Labels
    for x_var in xrange(n_traces - 1):
        hist_2d_axes[(x_var, n_traces-1)].set_xlabel(axis_labels[x_var],fontsize=20)

        hist_2d_axes[(x_var, n_traces-1)].xaxis.set_major_locator(MaxNLocator(4))
    for y_var in xrange(1, n_traces ):
        hist_2d_axes[(0,y_var)].set_ylabel(axis_labels[y_var],fontsize=20)
        hist_2d_axes[(0,y_var)].yaxis.set_major_locator(MaxNLocator(4))

    if fname != None:
        if len(fname.split('.')) == 1:
            fname += '.eps'
        plt.savefig(fname, transparent=True, bbox_inches = "tight")
    if display:
        plt.show()


def gus_contour(x,y,nbins=20,ncontours=10,log=False):
    """Make a basic contour plot from scattered data"""
    H,xedges,yedges = np.histogram2d(y,x,bins=nbins)
    extent = [yedges[0],yedges[-1],xedges[0],xedges[-1]]
    if not log: plt.contour(H,extent=extent,colors='k')
    else: plt.contour(H,extent=extent,colors='k',norm=LogNorm())
    return None
