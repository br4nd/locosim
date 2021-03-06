#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import dot, transpose, diag
from numpy.linalg import inv, norm
import pickle

import sys, os
sys.path.insert(0,os.path.abspath('..'))
from solvers.nls import nls_m1_a4_3d

import pdb

#-------------------------------------------------------------------------------
def set_config() :

    # Anchor locations
    #                x, y, z
    A = np.array([ [ 70,0,10],   # A0
                   [ 70,24,10],   # A1
                   [ 0,24,10],   # A2
                   [ 0,0,10] ]) # A3

    # Add random noise (row,col)
    #Rstd = [0.024, 0.055, 0.056, 0.239]
    rstd = [.007, .007, .007, .007]
    #Rbias = [-0.002, -0.002, -0.002, -0.002]  # range bias
    rbias = [0.0, 0.0, 0.0, 0.0]  # range bias  (negative makes range measurement shorter than truth)

    # x_vec = slice(-4, 4+dx, dx);
    # y_vec = slice(-4, 4+dy, dy);
    dx = 1   # pixel size in x dim
#    dx = 1
    xlim = [0,70]   # sim x start and stop locations
    nx = int((xlim[1]-xlim[0])/dx) + 1
    xvec = np.linspace(xlim[0],xlim[1],nx,endpoint=True)

    dy = 1    # pixel size in y dim
#    dy = 1
    ylim = [7,20.5]   # sim y start and stop locations
    ny = int((ylim[1]-ylim[0])/dy) + 1
    yvec = np.linspace(ylim[0],ylim[1],ny,endpoint=True)

    zt = 4.5    # sim plane altitude.
    # dz = 0.25
    # zlim = [0,0]
    # nz = int((zlim[1]-zlim[0])/dz) + 1
    # zvec = np.linspace(zlim[0],zlim[1],dz,endpoint=True)
    # if not zvec : # if empty
    #     zvec = 0

    ntrials = 5  # number of sims per spot

    config = { 'A': A,
               'xvec': xvec, 'yvec': yvec, 'zt': zt,
               'rstd': rstd, 'rbias': rbias,
               'ntrials': ntrials }

    return config

#-------------------------------------------------------------------------------
def monte_carlo_2d(config) :

    A = config['A']
    xvec = config['xvec']
    yvec = config['yvec']
    zt = config['zt']
    rstd = config['rstd']
    rbias = config['rbias']
    ntrials = config['ntrials']

    nx = len(xvec)
    ny = len(yvec)
    Emean = np.nan*np.ones((nx,ny))
    Estd = np.nan*np.ones((nx,ny))

    for ii in range(nx) :
        xt = xvec[ii]

        for jj in range(ny) :
            yt = yvec[jj]

            Mt = np.array([xt,yt,zt])  # x, y, z Mobile truth
            rtrue = norm(Mt-A,axis=1)  # true range

            Mh = np.nan*np.ones((ntrials,3))
            for kk in range(ntrials) :

                rnoise = rstd*np.random.randn(1,4)  # range error
                rmeas = rtrue + rbias + rnoise  # simulated range measurement

                Mh[kk],rc,residual = nls_m1_a4_3d(Mt,A,rmeas,rstd,max_residual=0.001,maxI=50)
                if rc != 0 :
                    print 'NLS Failure at (%3.2f,%3.2f,%3.2f)' % (Mt[0],Mt[1],Mt[2])
#                    pdb.set_trace()
                    pass

            Mh_mean = np.mean(Mh,axis=0)
            Emean[ii,jj] = norm(Mh_mean-Mt)
            Estd[ii,jj] = norm(np.std(Mh,axis=0))
            #print 'xt: %3.2f, yt: %3.2f' % (xt,yt)
            #print 'rc: % d, residual: %.5f' % (rc,residual)
            #print 'Mt_x: %3.5f, Mt_y: %3.5f, Mt_z: %3.5f' % (Mt[0],Mt[1],Mt[2])
            #print 'Mh_x: %3.5f, Mh_y: %3.5f, Mh_z: %3.5f' % (Mh_mean[0],Mh_mean[1],Mh_mean[2])
            #print 'E[%3.2f,%3.2f]: %3.7f' % (xt,yt,E[ii,jj])
        sys.stdout.write('.')
        sys.stdout.flush()
    print '\nDONE'

    return Emean, Estd

#-------------------------------------------------------------------------------
def store_data(config,Emean,Estd) :
    #pdb.set_trace()
    #D = {'A': A, 'xvec': xvec, 'yvec': yvec, 'E': E}
    fstore = open('data.pkl','wb')
    pickle.dump(config,fstore)
    pickle.dump(Emean,fstore)
    pickle.dump(Estd,fstore)
    fstore.close()

#-------------------------------------------------------------------------------
def load_data(fn) :
    fload = open(fn,'rb')
    config = pickle.load(fload)
    Emean = pickle.load(fload)
    Estd = pickle.load(fload)
    fload.close()
    return config, Emean, Estd


#-------------------------------------------------------------------------------
def plot_config(config) :
    A = config['A']
#    matplotlib.rcParams['legend.fontsize'] = 8

    # Create the plot
    fig = plt.figure(figsize=(7, 5))#, dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    #ax1 = fig.gca(projection='3d')
    ax.scatter(A[:,0],A[:,1],A[:,2],s=20,c='red',marker='d',edgecolors='red',label='Anchors')
    ax.set_zlim([0, 3])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    #ax1.legend()
    plt.draw()

#-------------------------------------------------------------------------------
def plot_results(xvec,yvec,Emean,Estd) :

    fig = plt.figure(figsize=(9, 12))#, dpi=150)

    ax = fig.add_subplot(211)
#    ax2 = fig.add_subplot(212)
#    pdb.set_trace()
#    Emean = Emean[:-1,:-1]
#    plt.pcolormesh(yvec,xvec,Emean,cmap='hot',shading='gouraud',vmin=0,vmax=Emean.max())
    plt.pcolormesh(yvec,xvec,Emean,cmap='jet',shading='gouraud',vmin=0,vmax=Emean.max())
    plt.imshow
    plt.title('pcolor')
    # set the limits of the plot to the limits of the data
    #plt.axis([xvec.min(), xvec.max(), yvec.min(), yvec.max()])
    plt.colorbar()

    ax2 = fig.add_subplot(212)
#    ax2 = fig.add_subplot(212)
    plt.pcolormesh(yvec,xvec,Estd,cmap='hot',shading='gouraud',vmin=0,vmax=Estd.max())
    plt.colorbar()

    plt.draw()

#-------------------------------------------------------------------------------
if __name__ == '__main__' :
    # TODO: testbenches for nls solvers

    config = set_config()
    
    import time
    t0 = time.time()
    Emean,Estd = monte_carlo_2d(config)
    t1 = time.time() - t0
    print 't1: %5.1f' % t1
    
    store_data(config,Emean,Estd)
	
    fn = 'data.pkl'
    config,Emean,Estd = load_data(fn)
    X,Y = np.meshgrid(config['xvec'],config['yvec'])
    Z = Estd
 
    from mpl_toolkits.mplot3d import axes3d
    plt.ion()
    fig = plt.figure(figsize=(7,5),dpi=150)
    ax = fig.add_subplot(111,projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    #ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0, cmap='hot',
            levels=np.linspace(0,Estd.max(),20))

    plt.suptitle('MonteCarlo n:%d, rstd:%3.3f, rbias:%3.3f'
        % (config['ntrials'], max(config['rstd']), max(config['rbias'])))

    #ax1 = fig.gca(projection='3d')
    A = config['A']
    ax.scatter(A[:,0],A[:,1],A[:,2],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')
    ax.set_zlim([0, 3])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')


    fig = plt.figure(figsize=(7,14),dpi=150)
    ax1 = fig.add_subplot(211)
    plt.pcolormesh(config['yvec'],config['xvec'],Emean,cmap='hot',vmin=0,vmax=np.abs(Emean).max())
    plt.suptitle('Plan View')
    plt.title('Emean')
    plt.colorbar()
    ax1.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax2 = fig.add_subplot(212)
    plt.pcolormesh(config['yvec'],config['xvec'],Estd,cmap='hot',vmin=0,vmax=Estd.max())
    plt.title('Estd')
    plt.colorbar()
    ax2.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')
    plt.suptitle('MonteCarlo n:%d, rstd:%3.3f, rbias:%3.3f'
        % (config['ntrials'], max(config['rstd']), max(config['rbias'])))



#    plt.show()


    pdb.set_trace()
