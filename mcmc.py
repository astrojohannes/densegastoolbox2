#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#from PTMCMCSampler import PTMCMCSampler
from scipy.spatial import distance
from scipy.interpolate import LinearNDInterpolator, griddata
import emcee
from multiprocessing import Pool

##################################################################

def find_nearest(models,values,cnt=1):

    mdls=np.vstack(models)

    distances=[]
    for i in range(len(models[0])):
        distances.append(distance.euclidean(mdls[:,i],values))

    if cnt==1:
        idx=np.argmin(distances)
    else:
        this_len=len(distances)
        if this_len>cnt:
            idx=np.argpartition(distances,cnt)[:cnt]
        else:
            idx=np.argpartition(distances,this_len-1)

    return idx

##################################################################

def getpos(labels,nwalkers,ndim,conf,sampler,nreps, nsims_burnin):
    ##### Define parameter grid for random selection of initial points for walker #######
    ##### PARAMETER GRID #####
    grid_n=1.9+np.arange(32)*0.1

    grid_T=conf.valid_T[1:]  # first value is 0
    grid_width=conf.valid_W[1:]  # first value is 0
    grid_tau_thin=[0.1,0.2,0.3]         # cross-check with calc_linerats.py
    grid_tau_middle=[0.8,1.1,1.5]           # cross-check with calc_linerats.py
    grid_tau_thick=[5.0,6.5,8.0]        # cross-check with calc_linerats.py

    grid_tau={}
    for ii,lbl in enumerate(labels[3:]):
        if lbl=='tau_12co': grid_tau[ii]=grid_tau_thick
        elif lbl=='tau_13co' or lbl=='tau_c18o' or lbl=='tau_c17o': grid_tau[ii]=grid_tau_thin
        else: grid_tau[ii]=grid_tau_middle

    if ndim==3:     # case tau is fixed
        pos = [np.array([ \
           np.random.choice(grid_n,size=1)[0],\
           np.random.choice(grid_T,size=1)[0],\
           np.random.choice(grid_width,size=1)[0]],\
           dtype=np.float64) for i in range(1)]
    else:   # case tau is free
		# messy solution for creating grid
        pos = np.empty((nwalkers, ndim), dtype = np.float64)
        pos[:,0] = np.random.choice(grid_n, size=nwalkers)
        pos[:,1] = np.random.choice(grid_T, size=nwalkers)
        pos[:,2] = np.random.choice(grid_width)
        for ii,lbl in enumerate(labels[3:]):
            pos[:,ii+3] = np.random.choice(grid_tau[ii], size=nwalkers)
    return pos

##################################################################

def getpos_old(labels,nwalkers,ndim,conf):
    ##### Define parameter grid for random selection of initial points for walker #######
    ##### PARAMETER GRID #####
    grid_n=1.9+np.arange(32)*0.1

    grid_T=conf.valid_T[1:]  # first value is 0
    grid_width=conf.valid_W[1:]  # first value is 0
    grid_tau_thin=[0.1,0.2,0.3]         # cross-check with calc_linerats.py
    grid_tau_middle=[0.8,1.1,1.5]           # cross-check with calc_linerats.py
    grid_tau_thick=[5.0,6.5,8.0]        # cross-check with calc_linerats.py

    grid_tau={}
    for ii,lbl in enumerate(labels[3:]):
        if lbl=='tau_12co': grid_tau[ii]=grid_tau_thick
        elif lbl=='tau_13co' or lbl=='tau_c18o' or lbl=='tau_c17o': grid_tau[ii]=grid_tau_thin
        else: grid_tau[ii]=grid_tau_middle

    if ndim==3:     # case tau is fixed
        pos = [np.array([ \
           np.random.choice(grid_n,size=1)[0],\
           np.random.choice(grid_T,size=1)[0],\
           np.random.choice(grid_width,size=1)[0]],\
           dtype=np.float64) for i in range(1)]
    else:   # case tau is free
        pos = [np.array([ \
           np.random.choice(grid_n,size=1)[0],\
           np.random.choice(grid_T,size=1)[0],\
           np.random.choice(grid_width,size=1)[0]]+\
           [np.random.choice(grid_tau[ii],size=1)[0] for ii,lbl in enumerate(labels[3:])],\
           dtype=np.float64) for i in range(1)]

    return pos[0]

##################################################################

def getprior(pp,grid_theta,ndim):

    tt=[]

    for ii in range(ndim):
        if np.all(np.min(grid_theta[ii]) < pp[ii]) and np.all(np.max(grid_theta[ii]) > pp[ii]):
            tt.append(0.0)
        else:
            tt.append(-np.inf)

    if -np.inf in tt:
        return -np.inf
    else:
        return 0.0

##################################################################

def getloglike(theta, grid_theta, grid_loglike, interp):

    intheta=np.array(theta,dtype=np.float64)
    diff=np.ones_like(grid_loglike)*1e20
    isclose=np.zeros_like(grid_loglike,dtype=bool)

    ###########################
    # nearest neighbor loglike
    if not interp:

        for i in range(len(grid_theta.T)):
            # calculate element-wise quadratic difference and sum it up
            # to get index of nearest neighbour on grid     
            diff[i]=((intheta-grid_theta.T[i])**2.0).sum()
            isclose[i]=np.allclose(intheta,grid_theta.T[i],rtol=0.5)

        # find nearest neighbour in multidim space
        ind=np.array(diff,dtype=np.float64).argmin()
        this_loglike=grid_loglike[ind]

        # check if the nearest neighbour is within some relative tolerance
        if not isclose[ind]:
            this_loglike = -np.inf

        if not np.isfinite(this_loglike):
            this_loglike = -np.inf

    #############################
    # interpolated loglike
    else:
        """
        cutout_idx=find_nearest(grid_theta,intheta,1000)
        grid_theta_cutout=np.array([x[cutout_idx] for x in grid_theta])
        grid_loglike_cutout=np.array(grid_loglike[cutout_idx])


        for i in range(len(grid_theta_cutout.T)):
            # calculate element-wise quadratic difference and sum it up
            # to get index of nearest neighbour on grid     
            diff[i]=((intheta-grid_theta_cutout.T[i])**2.0).sum()
            isclose[i]=np.allclose(intheta,grid_theta_cutout.T[i],rtol=3.0)

        if not isclose[cutout_idx[0]]:
            this_loglike = -np.inf

        else:
            # griddata with method='linear' sometimes fails with a Qhull error
            this_loglike = float(np.nan_to_num(griddata(grid_theta_cutout.T, grid_loglike_cutout, intheta, method='linear', rescale=False),nan=-np.inf))
        """
        
        this_interp = LinearNDInterpolator(grid_theta.T,grid_loglike,rescale=False)
        this_loglike = float(this_interp(intheta))

        if not np.isfinite(this_loglike):
            this_loglike = -np.inf

    return this_loglike

#######################################################################

def mymcmc(grid_theta, grid_loglike, ndim, nwalkers, interp, nsims, labels, conf, nreps, nsims_burnin, backend, n_cpus=1, pixelnr='1'):
    
    with Pool(processes=n_cpus) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, getloglike, args = ([grid_theta, grid_loglike, interp]), pool = pool, backend = backend)                                                                

        pos=getpos(labels,nwalkers,ndim, conf, sampler, nsims, nsims_burnin)

        for ii in range(nreps):
            # call positions and associated probabilities from sampler
            pos, prob, state = sampler.run_mcmc(pos, nsims_burnin, progress = True)
            # get highest prob position for walkers stuck in low probability space
            max_prob_index = np.argmax(prob)
            max_prob_pos = pos[max_prob_index,:]
        
            # mask for stuck walkers
            mask = prob == -np.inf        

            # initialize new walker positions in small ball around max probability
            # done individually due to different scales in parameters
            pos[mask,0] = max_prob_pos[0] + 1e3*np.random.randn(np.sum(mask))
            pos[mask,1] = max_prob_pos[1] + 1.5*np.random.randn(np.sum(mask))
            pos[mask,2] = max_prob_pos[2] + 1e-2*np.random.randn(np.sum(mask))
            # reset sampler
            sampler.reset()

        # do full sampling
        sampler.run_mcmc(pos, nsims, progress=True, store=True)

#######################################################################


def mymcmc_old(grid_theta, grid_loglike, ndim, nwalkers, interp, nsteps, labels, conf, pixelnr='1'):

    p0=getpos(labels,nwalkers,ndim, conf)

    # variance defines step_size
    step_size = 0.1
    cov = np.eye(ndim) * step_size**2

    sampler = PTMCMCSampler.PTSampler(ndim, getloglike, getprior, cov=np.copy(cov), loglargs=([grid_theta,grid_loglike,interp]), logpargs=([grid_theta,ndim]), outDir="./chains"+pixelnr)
    sampler.sample(p0, nsteps, burn=int(nsteps/5), thin=1, SCAMweight=10, AMweight=10, DEweight=10, NUTSweight=10, HMCweight=20, MALAweight=10, HMCsteps=50, HMCstepsize=0.08)

    return

