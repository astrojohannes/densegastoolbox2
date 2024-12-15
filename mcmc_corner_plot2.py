#!/usr/bin/env python

import numpy as np
import corner
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.optimize import minimize
import autocorr

############################################################

def mcmc_corner_plot(infile, outfile,labels,ndim,pixelnr=1):

    # adapt implementation from dgt v 1.7

    if len(labels)==3:  # case fixed optical depth
        print("[INFO] Optical depths are fixed")
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)]
    else:   # case free optical depth
        print("[INFO] Optical depths are free")
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)] + [(0.0,9.0) for x in range(len(labels[3:]))]



    reader = emcee.backends.HDFBackend(infile)

    tau = reader.get_autocorr_time(tol=0) # this tau is not optical depth, but the MCMC autocorrelation time

    for ii in range(ndim):
        print('[INFO] ACT for parameter'+str(ii+1)+':'+str(tau[ii]))

    tau_mean = np.mean(tau)
    print("[INFO] Mean autocorrelation time: {0:.3f} steps".format(tau_mean))
    print("[INFO] Mean acceptance fraction: {0:.3f}".format(np.mean(reader.accepted / reader.iteration)))


    if not pd.isna(np.nanmax(tau)) and not pd.isna(np.nanmin(tau)):
        burnin = int(2 * np.nanmax(tau))
        thin = int(0.5 * np.nanmin(tau))
        if thin == 0:
            thin = 1

        samples = reader.get_chain(flat=True, discard=burnin, thin=thin)
        logprob = reader.get_log_prob(flat=True, discard=burnin, thin=thin)

        nsteps = len(samples)
        print("[INFO] Number of steps is: ",nsteps)
        show_warning_nsteps = False
        for taui in tau:
            if nsteps < 50 * taui:
                show_warning_nsteps = True
        if show_warning_nsteps:
            print("[WARN] At least one tau value is >nsteps/50. You should consider to re-run a longer chain (increase nsteps). Your autocorrelation times (tau) are:",tau)

        logprob[logprob==-np.inf] = -1e40
        logprob[logprob==np.inf] = 1e40
        # check for convergence
        logprob_min=np.nanmin(logprob)
        logprob_max=np.nanmax(logprob)

        if logprob_max>logprob_min:
            converged=True
        else:
            converged=False
            print("[WARN] MCMC did not converge, you may try to increase the number of steps (nsteps)!")
            return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,[[np.nan,np.nan,np.nan,np.nan] for i in range(len(labels))]]


        #all_samples = np.concatenate((samples, logprob[:, None]), axis=1)
        all_samples = samples
        #labels += ["log prob"]

        # 0.16 and 0.84 percentiles correspond to +/- 1 sigma in a Gaussian
        q=[0.16, 0.5, 0.84]

        nicelabels=[]
        for label in labels:
            if label[0:3]=='tau': nicelabels.append('$\\tau_{'+label[4:]+'}$')
            else: nicelabels.append(label)

        figure=corner.corner(all_samples, labels=nicelabels,\
            range=myrange,\
            quantiles=q,\
            plot_datapoints=False,\
            plot_contours=True,\
            plot_density=True,\
            fill_contours=True,\
            contour_kwargs={'cmap':'viridis','colors':None},\
            contourf_kwargs={'cmap':'viridis','colors':None},\
            show_titles=True, title_kwargs={"fontsize": 16},\
            label_kwargs={"fontsize": 16}
        )

        # save corner plot
        figure.savefig(outfile,bbox_inches='tight')

        samples_n=samples[:,0]
        samples_T=samples[:,1]
        samples_W=samples[:,2]

        """
        plt.ion()
        figs={}
        for i in range(3):
            figs[i]=plt.figure()
            plt.clf()
            plt.hist(samples[:, i], 100, color="k", histtype="step")
            plt.pause(0.5)
            plt.xlabel(r"$\theta$ "+labels[i])
            plt.ylabel(r"$p(\theta)$ "+labels[i])
            plt.gca().set_yticks([]);
            plt.title("Histogram of samples")
        """

        # calculate quantile-based 1-sigma error bars
        q=[0.16,0.5,0.84]
        lowern_mcmc,bestn_mcmc,uppern_mcmc=np.quantile(samples_n,q)
        lowerT_mcmc,bestT_mcmc,upperT_mcmc=np.quantile(samples_T,q)
        lowerW_mcmc,bestW_mcmc,upperW_mcmc=np.quantile(samples_W,q)

        if len(labels)>3:
            taulist=[]
            for ii,label in enumerate(labels[3:]):
                """
                if label=='tau_12co': samples_CO=samples[:,ii+3]
                if label=='tau_13co': samples_13CO=samples[:,ii+3]
                if label=='tau_c17o': samples_C17O=samples[:,ii+3]
                if label=='tau_c18o': samples_C18O=samples[:,ii+3]
                if label=='tau_hcn': samples_HCN=samples[:,ii+3]
                if label=='tau_hnc': samples_HNC=samples[:,ii+3]
                if label=='tau_hcop': samples_HCOP=samples[:,ii+3]
                if label=='tau_cs': samples_CS=samples[:,ii+3]
                """

                if label[0:3]=='tau':
                    """
                    figs[ii+3]=plt.figure()
                    plt.clf()
                    plt.hist(samples[:, ii+3], 100, color="k", histtype="step")
                    plt.pause(0.5)
                    plt.xlabel(r"$\theta$ "+labels[ii+3])
                    plt.ylabel(r"$p(\theta)$ "+labels[ii+3])
                    plt.gca().set_yticks([]);
                    plt.title("Histogram of samples")
                    """

                    lowerTau_mcmc,bestTau_mcmc,upperTau_mcmc=np.quantile(samples[:,ii+3],q)
                    taulist.append([labels[ii+3],str(round(lowerTau_mcmc,2)),str(round(bestTau_mcmc,2)),str(round(upperTau_mcmc,2))])
        else:
            taulist=[np.nan,np.nan,np.nan,np.nan]

        bestn_mcmc_val=str(round(bestn_mcmc,2))
        bestn_mcmc_upper="+"+str(round(uppern_mcmc-bestn_mcmc,2))
        bestn_mcmc_lower="-"+str(round(bestn_mcmc-lowern_mcmc,2))

        bestT_mcmc_val=str(round(bestT_mcmc,2))
        bestT_mcmc_upper="+"+str(round(upperT_mcmc-bestT_mcmc,2))
        bestT_mcmc_lower="-"+str(round(bestT_mcmc-lowerT_mcmc,2))

        bestW_mcmc_val=str(round(bestW_mcmc,2))
        bestW_mcmc_upper="+"+str(round(upperW_mcmc-bestW_mcmc,2))
        bestW_mcmc_lower="-"+str(round(bestW_mcmc-lowerW_mcmc,2))

    else:
        print("[WARN] MCMC autocorrelation time (tau) is NaN. Did not converge and corner plot cannot be created!")
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,[[np.nan,np.nan,np.nan,np.nan] for i in range(len(labels))]]

    return [bestn_mcmc_val,bestn_mcmc_upper,bestn_mcmc_lower,bestT_mcmc_val,bestT_mcmc_upper,bestT_mcmc_lower,bestW_mcmc_val,bestW_mcmc_upper,bestW_mcmc_lower,taulist]


############################################################

def mcmc_corner_plot_old(outfile,labels,ndim,pixelnr=1):

    if len(labels)==3:  # case fixed optical depth
        print("[INFO] Optical depths are fixed")
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)]
    else:   # case free optical depth
        print("[INFO] Optical depths are free")
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)] + [(0.0,9.0) for x in range(len(labels[3:]))]

    chain = np.loadtxt('./chains'+pixelnr+'/chain_1.txt')
    # the last 4 columns are:
    # lnprob, lnlike, naccepted/iter, pt_acc

    samples = chain[:,:ndim]
    logprob = chain[:,ndim:ndim+1]

    nsteps=int(logprob.shape[0])

    print("[INFO] Number of steps is: ",nsteps)

    tau=[]
    for ii in range(ndim):
        this_sample=samples[:,ii:ii+1]
        this_sample=np.reshape(this_sample, this_sample.size)
        this_tau = autocorr.integrated_time(this_sample,quiet=True)   # this tau is not optical depth, but the MCMC autocorrelation time
        print('[INFO] ACT for parameter '+str(ii)+': '+str(this_tau))
        tau.append(this_tau)
    tau_mean = np.mean(tau)

    #print("[INFO] Mean acceptance fraction: {0:.3f}".format(np.mean(reader.accepted / reader.iteration)))
    print("[INFO] Mean autocorrelation time: {0:.3f} steps".format(tau_mean))

    show_warning_nsteps=False
    for taui in tau:
        if nsteps<50*taui: show_warning_nsteps=True

    if show_warning_nsteps:
        print("[WARN] At least one tau value is >nsteps/50. You should consider to re-run a longer chain (increase nsteps). Your autocorrelation times (tau) are:",tau)


    if not pd.isna(np.mean(tau)):

        # check for convergence
        logprob_min=np.nanmin(logprob)
        logprob_max=np.nanmax(logprob)
    
        if logprob_max>logprob_min:
            converged=True
        else:
            converged=False
            print("[WARN] MCMC did not converge, you may try to increase the number of steps (nsteps)!")
            return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,[[np.nan,np.nan,np.nan,np.nan] for i in range(len(labels))]]


        #all_samples = np.concatenate((samples, logprob[:, None]), axis=1)
        all_samples = samples
        #labels += ["log prob"]
    
        # 0.16 and 0.84 percentiles correspond to +/- 1 sigma in a Gaussian
        q=[0.16, 0.5, 0.84]

        nicelabels=[]
        for label in labels:
            if label[0:3]=='tau': nicelabels.append('$\\tau_{'+label[4:]+'}$')
            else: nicelabels.append(label)

        figure=corner.corner(all_samples, labels=nicelabels,\
            range=myrange,\
            quantiles=q,\
            plot_datapoints=False,\
            plot_contours=True,\
            plot_density=True,\
            fill_contours=True,\
            contour_kwargs={'cmap':'viridis','colors':None},\
            contourf_kwargs={'cmap':'viridis','colors':None},\
            show_titles=True, title_kwargs={"fontsize": 16},\
            label_kwargs={"fontsize": 16}
        )

        # save corner plot
        figure.savefig(outfile,bbox_inches='tight')

        samples_n=samples[:,0]
        samples_T=samples[:,1]
        samples_W=samples[:,2]

        """
        plt.ion()
        figs={}
        for i in range(3):
            figs[i]=plt.figure()
            plt.clf()
            plt.hist(samples[:, i], 100, color="k", histtype="step")
            plt.pause(0.5)
            plt.xlabel(r"$\theta$ "+labels[i])
            plt.ylabel(r"$p(\theta)$ "+labels[i])
            plt.gca().set_yticks([]);
            plt.title("Histogram of samples")
        """

        # calculate quantile-based 1-sigma error bars
        q=[0.16,0.5,0.84]
        lowern_mcmc,bestn_mcmc,uppern_mcmc=np.quantile(samples_n,q)
        lowerT_mcmc,bestT_mcmc,upperT_mcmc=np.quantile(samples_T,q)
        lowerW_mcmc,bestW_mcmc,upperW_mcmc=np.quantile(samples_W,q)

        if len(labels)>3:
            taulist=[]
            for ii,label in enumerate(labels[3:]):
                """
                if label=='tau_12co': samples_CO=samples[:,ii+3]
                if label=='tau_13co': samples_13CO=samples[:,ii+3]
                if label=='tau_c17o': samples_C17O=samples[:,ii+3]
                if label=='tau_c18o': samples_C18O=samples[:,ii+3]
                if label=='tau_hcn': samples_HCN=samples[:,ii+3]
                if label=='tau_hnc': samples_HNC=samples[:,ii+3]
                if label=='tau_hcop': samples_HCOP=samples[:,ii+3]
                if label=='tau_cs': samples_CS=samples[:,ii+3]
                """

                if label[0:3]=='tau':
                    """
                    figs[ii+3]=plt.figure()
                    plt.clf()
                    plt.hist(samples[:, ii+3], 100, color="k", histtype="step")
                    plt.pause(0.5)
                    plt.xlabel(r"$\theta$ "+labels[ii+3])
                    plt.ylabel(r"$p(\theta)$ "+labels[ii+3])
                    plt.gca().set_yticks([]);
                    plt.title("Histogram of samples")
                    """    

                    lowerTau_mcmc,bestTau_mcmc,upperTau_mcmc=np.quantile(samples[:,ii+3],q)
                    taulist.append([labels[ii+3],str(round(lowerTau_mcmc,2)),str(round(bestTau_mcmc,2)),str(round(upperTau_mcmc,2))])
        else:
            taulist=[np.nan,np.nan,np.nan,np.nan]    

        bestn_mcmc_val=str(round(bestn_mcmc,2))
        bestn_mcmc_upper="+"+str(round(uppern_mcmc-bestn_mcmc,2))
        bestn_mcmc_lower="-"+str(round(bestn_mcmc-lowern_mcmc,2))
    
        bestT_mcmc_val=str(round(bestT_mcmc,2))
        bestT_mcmc_upper="+"+str(round(upperT_mcmc-bestT_mcmc,2))
        bestT_mcmc_lower="-"+str(round(bestT_mcmc-lowerT_mcmc,2))
    
        bestW_mcmc_val=str(round(bestW_mcmc,2))
        bestW_mcmc_upper="+"+str(round(upperW_mcmc-bestW_mcmc,2))
        bestW_mcmc_lower="-"+str(round(bestW_mcmc-lowerW_mcmc,2))

    else:
        print("[WARN] MCMC autocorrelation time (tau) is NaN. Did not converge and corner plot cannot be created!")
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,[[np.nan,np.nan,np.nan,np.nan] for i in range(len(labels))]]

    return [bestn_mcmc_val,bestn_mcmc_upper,bestn_mcmc_lower,bestT_mcmc_val,bestT_mcmc_upper,bestT_mcmc_lower,bestW_mcmc_val,bestW_mcmc_upper,bestW_mcmc_lower,taulist]


