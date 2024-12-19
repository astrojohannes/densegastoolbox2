#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PTMCMCSampler import PTMCMCSampler
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
        pos[:,2] = np.random.choice(grid_width, size=nwalkers)
        for ii,lbl in enumerate(labels[3:]):
            pos[:,ii+3] = np.random.choice(grid_tau[ii], size=nwalkers)
    return pos

##################################################################

def getpos_ptmcmc(labels,nwalkers,ndim,conf):
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

    # Compute the bounds for each dimension based on the transposed grid
    min_bounds = grid_theta.T.min(axis=0)  # Minimum for each dimension
    max_bounds = grid_theta.T.max(axis=0)  # Maximum for each dimension

    ###########################
    # nearest neighbor loglike
    if not interp:

        for i in range(len(grid_theta.T)):
            # Calculate element-wise quadratic difference and sum it up
            diff[i] = ((intheta - grid_theta.T[i]) ** 2.0).sum()
   
            # Check if the current point is close and within bounds
            within_bounds = np.all((intheta >= min_bounds) & (intheta <= max_bounds))
            isclose[i] = np.allclose(intheta, grid_theta.T[i], rtol=0.5) and within_bounds
            #print(intheta,min_bounds,max_bounds,within_bounds)

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
        # Ensure theta is within bounds
        if not np.all((intheta >= min_bounds) & (intheta <= max_bounds)):
            return -np.inf  # Outside grid bounds

        # Find the indices of the neighbors along each axis
        neighbors = []
        for dim in range(grid_theta.shape[0]):
            axis_values = grid_theta[dim]
            lower_idx = np.searchsorted(axis_values, intheta[dim]) - 1
            upper_idx = lower_idx + 1
            if lower_idx < 0 or upper_idx >= len(axis_values):
                return -np.inf  # Out of bounds

            neighbors.append((lower_idx, upper_idx))

        # Perform linear interpolation
        interpolated_loglike = 0.0
        weights = np.ones(len(neighbors))  # Initialize weights
        for dim, (lower_idx, upper_idx) in enumerate(neighbors):
            lower_value = grid_theta[dim, lower_idx]
            upper_value = grid_theta[dim, upper_idx]
            lower_loglike = grid_loglike[lower_idx]
            upper_loglike = grid_loglike[upper_idx]

            # Linear interpolation weight
            weight_upper = (intheta[dim] - lower_value) / (upper_value - lower_value)
            weight_lower = 1.0 - weight_upper

            # Weighted sum
            interpolated_loglike += (
                weight_lower * lower_loglike + weight_upper * upper_loglike
            )

            #print(intheta,lower_value,lower_loglike,upper_value,upper_loglike)

        this_loglike = interpolated_loglike

        #print(this_loglike)
        #print()

    return this_loglike

#######################################################################


def mymcmc(grid_theta, grid_loglike, ndim, nwalkers, interp, nsims, labels, conf, nreps, nsims_burnin, backend, n_cpus=1, pixelnr='1', do_ptmcmc=False):

    if do_ptmcmc:

	    p0=getpos_ptmcmc(labels,nwalkers,ndim, conf)

	    # variance defines step_size
	    step_size = 0.1
	    cov = np.eye(ndim) * step_size**2

	    sampler = PTMCMCSampler.PTSampler(ndim, getloglike, getprior, cov=np.copy(cov), loglargs=([grid_theta,grid_loglike,interp]), logpargs=([grid_theta,ndim]), outDir="./chains"+pixelnr)
	    sampler.sample(p0, nsteps, burn=int(nsteps/5), thin=1, SCAMweight=10, AMweight=10, DEweight=10, NUTSweight=10, HMCweight=20, MALAweight=10, HMCsteps=50, HMCstepsize=0.08)

	    return

    else:
        with Pool(processes=n_cpus) as pool:

            moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, getloglike,
                args=([grid_theta, grid_loglike, interp]),
                moves=moves, pool=pool, backend=backend
            )

            pos = getpos(labels, nwalkers, ndim, conf, sampler, nsims, nsims_burnin)

            for ii in range(nreps):
                # Call positions and associated probabilities from sampler
                pos, prob, state = sampler.run_mcmc(pos, nsims_burnin, progress=True)

                # Get highest prob position for walkers stuck in low probability space
                max_prob_index = np.argmax(prob)
                max_prob_pos = pos[max_prob_index, :]

                # Mask for stuck walkers
                mask = prob == -np.inf
                print("mask sum: ", np.sum(mask))

                # Initialize new walker positions in small ball around max probability
                if np.sum(mask) != 0:
                    pos[mask, :] = max_prob_pos + 0.3 * np.random.randn(np.sum(mask), ndim)
                    pos[mask, 2] = max_prob_pos[2] + 0.1 * np.random.randn(np.sum(mask))

                # Reset sampler
                sampler.reset()

            # Do full sampling
            sampler.run_mcmc(pos, nsims, progress=True, store=True)

    return sampler


#######################################################################


