#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
import corner
import emcee
import matplotlib as mpl

if os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import autocorr


def _kde_mode_and_interval(values):
    """Return mode, +1sigma and -1sigma estimates robustly.

    Falls back to the median when KDE cannot be computed, e.g. for almost
    constant samples. This avoids UnboundLocalError/Singular-matrix failures
    on recent SciPy/NumPy versions.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    p16, median, p84 = np.percentile(values, [16, 50, 84])
    plus = p84 - median
    minus = median - p16

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return median, plus, minus, p84, p16

    kde_x = np.linspace(vmin, vmax, 1000)
    try:
        kde = gaussian_kde(values)
        kde_y = kde(kde_x)
        mode = kde_x[np.nanargmax(kde_y)]
    except Exception:
        mode = median

    return mode, plus, minus, p84, p16

############################################################

def mcmc_corner_plot(infile, outfile,labels,ndim,pixelnr=1):

    if len(labels)==3:  # case fixed optical depth
        print("[INFO] Optical depths are fixed")
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)]
    else:   # case free optical depth
        print("[INFO] Optical depths are free")
        #myrange=[(1.8,5.1),(8,32),(0.1,0.9)] + [(0.0,9.0) for x in range(len(labels[3:]))]

        myrange=[(1.8,5.1),(8,32),(0.1,0.9)]
        myrange_tau=[]
        for lbl in labels[3:]:
            if lbl.startswith("tau_12co"):
                range_tau = (4.5,8.5)
            elif lbl.startswith("tau_c18o") or lbl.startswith("tau_c17o") or lbl.startswith("tau_13co"):
                range_tau = (0.05,0.35)
            else:
                range_tau = (0.7,1.6)

            myrange_tau.append(range_tau)

        myrange += myrange_tau

    reader = emcee.backends.HDFBackend(infile)
    tau = reader.get_autocorr_time(tol=0) # this tau is not optical depth, but the MCMC autocorrelation time

    for ii in range(ndim):
        print('[INFO] ACT for parameter'+str(ii+1)+':'+str(tau[ii]))

    tau_mean = np.mean(tau)
    print("[INFO] Mean autocorrelation time: {0:.3f} steps".format(tau_mean))
    print("[INFO] Mean acceptance fraction: {0:.3f}".format(np.mean(reader.accepted / reader.iteration)))


    if not pd.isna(np.nanmax(tau)) and not pd.isna(np.nanmin(tau)):
        burnin = int(4 * np.nanmax(tau))
        thin = max(1, int(0.5 * np.nanmin(tau)))
        thin = 1

        nsteps=reader.get_chain(flat=False, discard=burnin, thin=thin).shape[0]
        nwalkers=reader.get_chain(flat=False, discard=burnin, thin=thin).shape[1]

        samples = reader.get_chain(flat=True, discard=burnin, thin=thin)
        logprob = reader.get_log_prob(flat=True, discard=burnin, thin=thin)

        print("[INFO] Number of steps is: ",nsteps)
        print("[INFO] Number of walkers is: ",nwalkers)

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

        n_params = all_samples.shape[1]

        # Initialize lists for best-fit values
        kde_modes = []
        kde_maxy = []
        uncertainty_pos = []
        uncertainty_neg = []

        # lists for results to return
        result = []
        taulist = []
        nicelabels=[]
        for i,y in enumerate(labels):
            label = labels[i]
            if label[0:3]=='tau':
                nicelabels.append('$\\tau_{'+label[4:]+'}$')
                is_tauval = True
            else:
                nicelabels.append(label)
                is_tauval = False

            kde_mode, upper_bound_1sigma, lower_bound_1sigma, p84, p16 = _kde_mode_and_interval(all_samples[:, i])
            uncertainty_pos.append(p84)
            uncertainty_neg.append(p16)
            kde_modes.append(kde_mode)
            kde_maxy.append(np.nan)

            if not is_tauval:
                result.append(round(kde_mode, 2) if np.isfinite(kde_mode) else np.nan)
                result.append(round(upper_bound_1sigma, 2) if np.isfinite(upper_bound_1sigma) else np.nan)
                result.append(round(lower_bound_1sigma, 2) if np.isfinite(lower_bound_1sigma) else np.nan)
            else:
                taulist.append([
                    label,
                    round(kde_mode, 2) if np.isfinite(kde_mode) else np.nan,
                    round(upper_bound_1sigma, 2) if np.isfinite(upper_bound_1sigma) else np.nan,
                    round(lower_bound_1sigma, 2) if np.isfinite(lower_bound_1sigma) else np.nan,
                ])


        figure=corner.corner(all_samples, labels=nicelabels,\
            range=myrange,\
            #quantiles=q,\
            plot_datapoints=False,\
            plot_contours=True,\
            plot_density=True,\
            fill_contours=True,\
            #contour_kwargs={'cmap':'viridis','colors':None},\
            #contourf_kwargs={'cmap':'viridis','colors':None},\
            show_titles=True, title_kwargs={"fontsize": 16},\
            label_kwargs={"fontsize": 16},
            levels=[0.1,0.3,0.6,0.9]
        )

        # Add vertical lines for the best-fit values on each histogram and plot Gaussian
        axes = np.array(figure.axes).reshape((n_params, n_params))

        for i in range(n_params):
            if np.isfinite(kde_modes[i]):
                ax = axes[i, i]  # Access the histograms along the diagonal

                # Plot Best-Fit and uncertainties as vertical lines
                ax.axvline(kde_modes[i], color='black', linestyle='-', label=f'best fit')
                ax.axvline(uncertainty_pos[i], color='red', linestyle='--',label='uncertainty (+)')
                ax.axvline(uncertainty_neg[i], color='red', linestyle='--',label='uncertainty (-)')

                # Add a legend for the first plot
                if i == 0:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # save corner plot
        figure.savefig(outfile,bbox_inches='tight')
        plt.close(figure)

    else:
        print("[WARN] MCMC autocorrelation time (tau) is NaN. Did not converge and corner plot cannot be created!")
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,[[np.nan,np.nan,np.nan,np.nan] for i in range(len(labels))]]

    r = result + [taulist]
    return r


############################################################

def mcmc_corner_plot_ptmcmc(outfile,labels,ndim,pixelnr=1):

    if len(labels)==3:  # case fixed optical depth
        print("[INFO] Optical depths are fixed")
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)]
    else:   # case free optical depth
        print("[INFO] Optical depths are free")
        #myrange=[(1.8,5.1),(8,32),(0.1,0.9)] + [(0.0,9.0) for x in range(len(labels[3:]))]
        myrange=[(1.8,5.1),(8,32),(0.1,0.9)]
        myrange_tau=[]

        for lbl in labels[3:]:
            if lbl.startswith("tau_12co"):
                range_tau = (4.5,8.5)
            elif lbl.startswith("tau_c18o") or lbl.startswith("tau_c17o") or lbl.startswith("tau_13co"):
                range_tau = (0.05,0.35)
            else:
                range_tau = (0.7,1.6)

            myrange_tau.append(range_tau)

        myrange += myrange_tau

    chain = np.loadtxt(Path('./chains' + str(pixelnr)) / 'chain_1.txt')
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
        plt.close(figure)

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


