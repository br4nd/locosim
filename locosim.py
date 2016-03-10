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
    # TODO read from .csv file

    # Anchor locations
    #                x, y, z
    A = np.array([ [ 4,-5, 1],   # A0
                   [ 4, 4, 2],   # A1
                   [-5, 4, 1],   # A2
                   [-4,-4, 2] ]) # A3

    # Add random noise (row,col)
    #Rstd = [0.024, 0.055, 0.056, 0.239]
    rstd = [.025, .025, .025, .025]
    #Rbias = [-0.002, -0.002, -0.002, -0.002]  # range bias
    rbias = [0.01, 0.01, 0.01, 0.01]  # range bias

    # x_vec = slice(-4, 4+dx, dx);
    # y_vec = slice(-4, 4+dy, dy);
    dx = 0.5
    #dx = 1
    xlim = [-15,15]
    nx = int((xlim[1]-xlim[0])/dx) + 1
    xvec = np.linspace(xlim[0],xlim[1],nx,endpoint=True)

    dy = 0.5
    #dy = 1
    ylim = [-7,7]
    ny = int((ylim[1]-ylim[0])/dy) + 1
    yvec = np.linspace(ylim[0],ylim[1],ny,endpoint=True)

    zt = -1.0
    # dz = 0.25
    # zlim = [0,0]
    # nz = int((zlim[1]-zlim[0])/dz) + 1
    # zvec = np.linspace(zlim[0],zlim[1],dz,endpoint=True)
    # if not zvec : # if empty
    #     zvec = 0

    ntrials = 5

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
    Emean_x = np.nan*np.ones((nx,ny))
    Estd_x = np.nan*np.ones((nx,ny))
    Emean_z = np.nan*np.ones((nx,ny))
    Estd_z = np.nan*np.ones((nx,ny))

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

                Mh[kk],rc,residual = nls_m1_a4_3d(Mt,A,rmeas,rstd,max_residual=0.01,maxI=50)
                if rc != 0 :
                    print 'NLS max at (%3.2f,%3.2f,%3.2f), residual=%5.5f'\
                        % (Mt[0], Mt[1], Mt[2], residual)
                    pass

            Mh_mean = np.mean(Mh,axis=0)
            Emean[ii,jj] = norm(Mh_mean-Mt)
            Estd[ii,jj] = norm(np.std(Mh,axis=0))
            #pdb.set_trace()
            Emean_x[ii,jj] = norm(Mh_mean[0]-Mt[0])
            Estd_x[ii,jj] = np.std(Mh[:,0])
            Emean_z[ii,jj] = norm(Mh_mean[2]-Mt[2])
            Estd_z[ii,jj] = np.std(Mh[:,2])
            #print 'xt: %3.2f, yt: %3.2f' % (xt,yt)
            #print 'rc: % d, residual: %.5f' % (rc,residual)
            #print 'Mt_x: %3.5f, Mt_y: %3.5f, Mt_z: %3.5f' % (Mt[0],Mt[1],Mt[2])
            #print 'Mh_x: %3.5f, Mh_y: %3.5f, Mh_z: %3.5f' % (Mh_mean[0],Mh_mean[1],Mh_mean[2])
            #print 'E[%3.2f,%3.2f]: %3.7f' % (xt,yt,E[ii,jj])
        sys.stdout.write('.')
        sys.stdout.flush()
    print '\nDONE'

    D = {'Emean': Emean, 'Estd': Estd,
        'Emean_x': Emean_x, 'Estd_x': Estd_x,
        'Emean_z': Emean_z, 'Estd_z': Estd_z}
    return D

#-------------------------------------------------------------------------------
def store_data(fn,config,D) :
    #pdb.set_trace()
    #D = {'A': A, 'xvec': xvec, 'yvec': yvec, 'E': E}
    fstore = open(fn,'wb')
    pickle.dump(config,fstore)
    pickle.dump(D,fstore)
    fstore.close()

#-------------------------------------------------------------------------------
def load_data(fn) :
    fload = open(fn,'rb')
    config = pickle.load(fload)
    D = pickle.load(fload)
    fload.close()
    return config, D


#-------------------------------------------------------------------------------
if __name__ == '__main__' :
    import time
    t0 = time.time()

    # comment this line to run monte carlo.  read stored otherwise
    #fn = 'test.pkl'
    if not 'fn' in locals() :
        ####################  Read config parameters
        # TODO: prompt user for ilename.csv file
        # or, if .pkl selected, skip over sim and read/plot stored data
        config = set_config()

        ####################  Run sim
        D = monte_carlo_2d(config)
        t1 = time.time() - t0
        print 't1: %5.2f' % t1

        # TODO try gdop analysis

        ####################  Store sim results
        # TODO: store in filename.pkl
        fn = 'test.pkl'
        store_data(fn,config,D)

    config,D = load_data(fn)
    Emean = np.transpose(D['Emean'])
    Estd = np.transpose(D['Estd'])
    Emean_x = np.transpose(D['Emean_x'])
    Estd_x = np.transpose(D['Estd_x'])
    Emean_z = np.transpose(D['Emean_z'])
    Estd_z = np.transpose(D['Estd_z'])

    X,Y = np.meshgrid(config['xvec'],config['yvec'])

    ####################  Plot sim results - 3D
    from mpl_toolkits.mplot3d import axes3d
    myshading = 'gourard'
    myshading = 'none'
    plt.ion()
#    fig = plt.figure(figsize=(7,5),dpi=150)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    #ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    #pdb.set_trace()
    cset = ax.contourf(X, Y, Estd, zdir='z', offset=zt, cmap='hot',
            levels=np.linspace(0,Estd.max(),100))

    plt.suptitle('MonteCarlo n:%d, rstd:%3.3f, rbias:%3.3f'
        % (config['ntrials'], max(config['rstd']), max(config['rbias'])))

    #ax1 = fig.gca(projection='3d')
    A = config['A']
    ax.scatter(A[:,0],A[:,1],A[:,2],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')
    ax.set_zlim([0, 3])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
#    ax.axis_equal(True)

    ####################  Plot sim results - Estd
#    fig = plt.figure(figsize=(7,14),dpi=150)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean,cmap='hot',shading=myshading,vmin=0,vmax=np.abs(Emean).max())
    #plt.suptitle('Plan View')
    plt.title('Emean')
    plt.colorbar()
    ax1.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')

    ax2 = fig.add_subplot(212)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd,cmap='hot',shading=myshading,vmin=0,vmax=Estd.max())
    plt.title('Estd')
    plt.colorbar()
    ax2.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')
    plt.suptitle('MonteCarlo n:%d, rstd:%3.3f, rbias:%3.3f'
        % (config['ntrials'], max(config['rstd']), max(config['rbias'])))#

    # Add figure for x, y, and z errors
    maxv = np.max([Emean_x.max(),Estd_x.max(),Emean_z.max(),Estd_z.max()])
    fig = plt.figure()

    ax1 = fig.add_subplot(321)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean_x,cmap='hot',shading=myshading,vmin=0,vmax=np.abs(Emean_x).max())
    plt.title('Emean_x, max=%3.3f' % np.abs(Emean_x).max())
    plt.colorbar()
    ax1.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')

    ax2 = fig.add_subplot(322)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd_x,cmap='hot',shading=myshading,vmin=0,vmax=np.abs(Estd_x).max())
    plt.title('Estd_x, max=%3.3f' % np.abs(Estd_x).max())
    plt.colorbar()
    ax2.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')

    ax = fig.add_subplot(325)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean_z,cmap='hot',shading=myshading,vmin=0,vmax=np.abs(Emean_z).max())
    plt.title('Emean_z, max=%3.3f' % np.abs(Emean_z).max())
    plt.colorbar()
    ax.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')

    ax = fig.add_subplot(326)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd_z,cmap='hot',shading=myshading,vmin=0,vmax=np.abs(Estd_z).max())
    plt.title('Estd_z, max=%3.3f' % np.abs(Estd_z).max())
    plt.colorbar()
    ax.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='lightblue',label='Anchors')

    # TODO: store as pdf


#    plt.show()


    t2 = time.time() - t0
    print 't2: %5.2f' % t2

    pdb.set_trace()
