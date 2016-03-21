#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
>>> python locosim.py

prompts user to choose parameters.csv in a folder
creates saved.pkl and plots.pdf in the same folder

Originally created Feb 2016
@author: B. Dewberry
Feb 21 2016: changed name to qualify.py, added qualify-info.csv
Version 2, Mar 20 2016 :
- fixed geometry inversion bug
- queries user for <parameters>.csv file
- reads config parameters from <parameters>.csv file in any folder
- stores saved.pkl in same folder as <parameters>.csv
- stores plots.pkl in same folder as <parameters>.csv
"""
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy import dot, transpose, diag
from numpy.linalg import inv, norm
import pickle

import sys, os
sys.path.insert(0,os.path.abspath('..'))
from solvers.nls import nls_m1_a4_3d

import pdb, csv
from pprint import pprint as pp


config_file = 'locosim_input_parameters.csv'

#-------------------------------------------------------------------------------
def get_config(config_file) :
    # TODO read from .csv file

    if config_file == 'test' :
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

        blur = 'none'

        ntrials = 5

        config = { 'A': A,
                   'xvec': xvec,
                   'yvec': yvec,
                   'zt': zt,
                   'rstd': rstd,
                   'rbias': rbias,
                   'ntrials': ntrials,
                   'blur' : blur }

        return config

    elif os.path.isfile(config_file) :
        #--- read config from file
        print 'Reading config parameters from %s.' % config_file
        A = []

        with open(config_file,'rb') as csvfile :
            #    stuff = csv.reader(csvfile,delimiter=' ',quotechar='|')
            reader = csv.reader(csvfile)
            for row in reader :
                #print ', '.join(row)
                if row[0] == '' :
                    continue
                if 'A' == row[0][0] :
                    if 'Anchors' == row[0] :
                        continue
                    A.append([ float(row[1]), float(row[2]), float(row[3]) ])
                elif 'rstd' == row[0] :
                    rstd = map(float,row[1:])
                elif 'rbias' == row[0] :
                    rbias = map(float,row[1:])
                elif 'dx' == row[0] :
                    dx = float(row[1])
                elif 'dy' == row[0] :
                    dy = float(row[1])
                elif 'xlim' == row[0] :
                    xlim = map(float,row[1:3])
                elif 'ylim' == row[0] :
                    ylim = map(float,row[1:3])
                elif 'zt' == row[0] :
                    zt = float(row[1])
                elif 'ntrials' == row[0] :
                    ntrials = int(row[1])
#                elif 'blur' == row[0] :
#                    blur = row[1]

            nx = int((xlim[1]-xlim[0])/dx) + 1
            xvec = np.linspace(xlim[0],xlim[1],nx,endpoint=True)
            ny = int((ylim[1]-ylim[0])/dy) + 1
            yvec = np.linspace(ylim[0],ylim[1],ny,endpoint=True)

            config = { 'A': np.array(A),
                       'xvec': xvec,
                       'yvec': yvec,
                       'zt': zt,
                       'rstd': rstd,
                       'rbias': rbias,
                       'ntrials': ntrials,
#                       'blur' : blur }
                      }
            pp(config)

            return config

    #pdb.set_trace()

    else :
        raise ValueError('No config file found named %s.' % config_file)

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
    Emean_y = np.nan*np.ones((nx,ny))
    Estd_y = np.nan*np.ones((nx,ny))
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
            Emean_y[ii,jj] = norm(Mh_mean[1]-Mt[1])
            Estd_y[ii,jj] = np.std(Mh[:,1])
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
        'Emean_y': Emean_y, 'Estd_y': Estd_y,
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

    ffn_config = []
    #ffn_config = './test2/parameters.csv'

    if not ffn_config :
        import Tkinter, tkFileDialog
        prompt = 'Choose an config file'
        types = [('All files','*'),('csv','*.csv')]
        defaultextension = '*'
        root = Tkinter.Tk()
        root.withdraw() # don't want a full GUI
        root.update()
        ffn_config = tkFileDialog.askopenfilename(parent=root,title=prompt,filetypes=types)

    path,fn_config = os.path.split(ffn_config)
    fn_main,fn_ext = os.path.splitext(fn_config)
    fn_pkl = 'saved.pkl'
    ffn_pkl = os.path.join(path,fn_pkl)

    fn_pdf = 'plots.pdf'
    ffn_pdf = os.path.join(path,fn_pdf)
    pdf = PdfPages(ffn_pdf)

    #--- Read config parameters
    #config = get_config('test')
    config = get_config(ffn_config)

    #------------------  Run sim
    D = monte_carlo_2d(config)
    t1 = time.time() - t0

    # TODO try gdop analysis

    #------------------- Store sim results
    store_data(ffn_pkl,config,D)

    #------------------- Compute stats
    config,D = load_data(ffn_pkl)
    Emean = np.transpose(D['Emean'])
    Estd = np.transpose(D['Estd'])
    Emean_x = np.transpose(D['Emean_x'])
    Estd_x = np.transpose(D['Estd_x'])
    Emean_y = np.transpose(D['Emean_y'])
    Estd_y = np.transpose(D['Estd_y'])
    Emean_z = np.transpose(D['Emean_z'])
    Estd_z = np.transpose(D['Estd_z'])

    #-------------------- Plot stuff
    X,Y = np.meshgrid(config['xvec'],config['yvec'])

    from mpl_toolkits.mplot3d import axes3d
    #shading = 'gourard'
    shading = 'none'

    plt.ion()
    #---------------  Plot sim results - 3D
#    fig = plt.figure(figsize=(7,5),dpi=150)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.05)
    #ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

    zt = config['zt']
    cset = ax.contourf(X, Y, Estd, zdir='z', offset=zt, cmap='hot',shading=shading,levels=np.linspace(0,Estd.max(),100))

    plt.suptitle('MonteCarlo n:%d, rstd:%.3f, rbias:%.3f'
        % (config['ntrials'], max(config['rstd']), max(config['rbias'])))
    A = config['A']
    ax.scatter(A[:,0],A[:,1],A[:,2],s=25,c='blue',marker='d',edgecolors='blue',label='Anchors')
    ax.set_zlim([0, max(A[:,2])+1])

    #-------------------- Plot slant range errors
    #fig = plt.figure(figsize=(7,14),dpi=150)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Emean).max())
    plt.title('Ebias (max=%.7f)' % np.max(Emean))
    plt.colorbar()
    ax1.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax2 = fig.add_subplot(212)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd,cmap='hot',shading=shading,vmin=0,vmax=Estd.max())
    plt.title('Estd (max=%.7f)' % np.max(Estd))
    plt.colorbar()
    ax2.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

#    plt.suptitle('MonteCarlo n:%d, max(emean):%.3fcm, max(estd):%.3fcm'
#        % (config['ntrials'], np.max(Emean)*100., np.max(Estd)*100.))#

    #-------------------- Plot sim results - Estd
    maxv = np.max([Emean_x.max(),Estd_x.max(),Emean_z.max(),Estd_z.max()])
    fig = plt.figure()

    ax1 = fig.add_subplot(321)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean_x,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Emean_x).max())
    #pdb.set_trace()
    plt.title('Emean_x, max=%.7f' % np.abs(Emean_x).max())
    plt.colorbar()
    ax1.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax2 = fig.add_subplot(322)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd_x,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Estd_x).max())
    plt.title('Estd_x, max=%.7f' % np.abs(Estd_x).max())
    plt.colorbar()
    ax2.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax3 = fig.add_subplot(323)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean_y,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Emean_y).max())
    plt.title('Emean_y, max=%.7f' % np.abs(Emean_y).max())
    plt.colorbar()
    ax3.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax4 = fig.add_subplot(324)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd_y,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Estd_y).max())
    plt.title('Estd_y, max=%.7f' % np.abs(Estd_y).max())
    plt.colorbar()
    ax4.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax5 = fig.add_subplot(325)
    plt.pcolormesh(config['xvec'],config['yvec'],Emean_z,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Emean_z).max())
    plt.title('Emean_z, max=%.7f' % np.abs(Emean_z).max())
    plt.colorbar()
    ax5.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    ax6 = fig.add_subplot(326)
    plt.pcolormesh(config['xvec'],config['yvec'],Estd_z,cmap='hot',shading=shading,vmin=0,vmax=np.abs(Estd_z).max())
    plt.title('Estd_z, max=%.7f' % np.abs(Estd_z).max())
    plt.colorbar()
    ax6.scatter(A[:,0],A[:,1],s=20,c='blue',marker='d',edgecolors='blue',label='Anchors')

    # TODO: store as pdf

#    plt.show()

    print '  compute time: %5.3f' % t1
    print '  total time: %5.3f' % (time.time() - t0)

    pdb.set_trace()
