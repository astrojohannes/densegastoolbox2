#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PTMCMCSampler import PTMCMCSampler
from scipy.spatial import distance
from scipy.interpolate import LinearNDInterpolator, griddata
import emcee
from multiprocessing import Pool
import itertools

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

    # Compute the bounds for each dimension based on the transposed grid
    min_bounds = grid_theta.T.min(axis=0)  # Minimum for each dimension
    max_bounds = grid_theta.T.max(axis=0)  # Maximum for each dimension

    # Ensure intheta is within bounds
    #print(intheta,min_bounds,max_bounds)
    if not np.all((intheta >= min_bounds) & (intheta <= max_bounds)):
        #print("Intheta",intheta,"outside bounds. Returning -np.inf")
        return -np.inf  # Outside grid bounds


    ###########################
    # nearest neighbor loglike
    if not interp:

        # Calculate squared differences for all grid points (vectorized)
        diff = np.sum((grid_theta.T - intheta) ** 2, axis=1)

        # Find the index of the nearest neighbor
        nearest_idx = np.argmin(diff)

        this_loglike=grid_loglike[nearest_idx]

        if not np.isfinite(this_loglike):
            return -np.inf


    #############################
    # interpolated loglike
    else:
        """
        # Find the indices of neighbors along each dimension
        indices = []
        for dim in range(n_params):
            axis_values = grid_theta[dim]  # Values along this dimension/parameter
            sorted_indices = np.argsort(axis_values)  # Sort indices for easier neighbor lookup
            axis_values_sorted = axis_values[sorted_indices]

            # Find lower and upper neighbors
            lower_idx = np.searchsorted(axis_values_sorted, intheta[dim]) - 1
            upper_idx = lower_idx + 1

            print(intheta,dim, axis_values_sorted[lower_idx], axis_values_sorted[upper_idx], grid_loglike[sorted_indices[lower_idx]], grid_loglike[sorted_indices[upper_idx]])

            if lower_idx < 0 or upper_idx >= n_points:
                #print("Intheta",intheta,"too close to bounds for interpolation. Returning -np.inf")
                return -np.inf  # Out of bounds

            indices.append([sorted_indices[lower_idx], sorted_indices[upper_idx]])

        # Generate all combinations of neighbors
        neighbor_combinations = list(itertools.product(*indices))  # Shape: (2^n, n)

        # Interpolate loglike using all neighbors
        interpolated_loglike = 0.0
        total_weight = 0.0

        for combination in neighbor_combinations:
            weight = 1.0
            for dim, idx in enumerate(combination):
                axis_values = grid_theta[dim]
                lower_value = axis_values[indices[dim][0]]
                upper_value = axis_values[indices[dim][1]]

                if axis_values[idx] < intheta[dim]:
                    weight *= (intheta[dim] - lower_value) / (upper_value - lower_value)
                else:
                    weight *= (upper_value - intheta[dim]) / (upper_value - lower_value)


            # Compute flat index for 1D grid_loglike
            flat_index = combination[0]  # Combination gives direct index in grid_loglike
            interpolated_loglike += weight * grid_loglike[flat_index]
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            interpolated_loglike /= total_weight
        else:
            print("Total weight is 0 or less. Returning -np.inf.")
            interpolated_loglike = -np.inf

        this_loglike = interpolated_loglike
        print("Final loglike for theta ",intheta," is ",str(this_loglike))
        print()
        """

        n_params, n_points = grid_theta.shape

        # Find the indices of neighbors along each dimension
        indices = []
        for dim in range(n_params):
            axis_values = grid_theta[dim]  # Values along this dimension/parameter
            sorted_indices = np.argsort(axis_values)  # Sort indices for easier neighbor lookup
            axis_values_sorted = axis_values[sorted_indices]

            # Find lower and upper neighbors
            lower_idx = np.searchsorted(axis_values_sorted, intheta[dim]) - 1
            upper_idx = lower_idx + 1

            #print(intheta, dim, axis_values_sorted[lower_idx], axis_values_sorted[upper_idx], grid_loglike[sorted_indices[lower_idx]], grid_loglike[sorted_indices[upper_idx]])

            if lower_idx < 0 or upper_idx >= n_points:
                return -np.inf  # Out of bounds

            indices.append([sorted_indices[lower_idx], sorted_indices[upper_idx]])

        # Generate all combinations of neighbors
        neighbor_combinations = list(itertools.product(*indices))  # Shape: (2^n, n)

        # Interpolate loglike using all neighbors (simple average)
        interpolated_loglike = 0.0

        for combination in neighbor_combinations:
            # Compute flat index for 1D grid_loglike
            flat_index = combination[0]  # Combination gives direct index in grid_loglike
            interpolated_loglike += grid_loglike[flat_index]

        # Normalize by the number of neighbors (simple averaging)
        interpolated_loglike /= len(neighbor_combinations)

        this_loglike = interpolated_loglike
        #print("Final loglike for theta ", intheta, " is ", str(this_loglike))
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

            moves_test1 = [(emcee.moves.DEMove(sigma=1e-9,gamma0=0.005), 0.3), (emcee.moves.DEMove(sigma=1e-7,gamma0=0.02), 0.3), (emcee.moves.DESnookerMove(gammas=0.5), 0.3), (emcee.moves.KDEMove(), 0.1)]
            moves_test2 = [ (emcee.moves.DEMove(sigma=1e-9,gamma0=0.0025), 0.2), \
                            (emcee.moves.DEMove(sigma=1e-8,gamma0=0.01), 0.2), \
                            (emcee.moves.DEMove(sigma=1e-7,gamma0=0.04), 0.2), \
                            (emcee.moves.DESnookerMove(gammas=0.5),0.3), \
                            (emcee.moves.KDEMove(), 0.1)]

            moves = moves_test2

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, getloglike,
                args=([grid_theta, grid_loglike, interp]),
                moves = moves,
                pool=pool, backend=backend
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
                print("Number of walkers stuck at -np.inf: ", np.sum(mask))

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


