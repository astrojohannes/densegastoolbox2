#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################################
# This takes observed intensities I of multiple lines from a table, e.g.
# 'ascii_galaxy.txt' and uses line ratios to perform a chi2 test on a
# radiative transfer model grid. The relative abundances (to CO) are fixed.
# The main result is a new table, e.g. 'ascii_galaxy_nT.txt'.
######################################################################################

import os
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
from matplotlib import rc
from scipy.interpolate import Rbf
from scipy.stats import chi2 as scipychi2
from scipy.spatial import distance
from pylab import *
from read_grid_ndist2 import read_grid_ndist,linename_obs2mdl,linename_mdl2obs
import mcmc
from datetime import datetime
import warnings
from mcmc_corner_plot2 import mcmc_corner_plot, mcmc_corner_plot_ptmcmc
import emcee

mpl.use("TkAgg")

cmap='cubehelix'

DEBUG=False

# ignore some warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered") 
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="overflow encountered in power")


##################################################################

mpl.rc('lines', linewidth=3)
mpl.rc('axes', linewidth=2)
mpl.rc('xtick.major', size=4)
mpl.rc('ytick.major', size=4)
mpl.rc('xtick.minor', size=2)
mpl.rc('ytick.minor', size=2)
mpl.rc('axes', grid=False)
mpl.rc('xtick.major', width=1)
mpl.rc('xtick.minor', width=1)
mpl.rc('ytick.major', width=1)
mpl.rc('ytick.minor', width=1)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

##################################################################

def find_nearest(models,values):

    mdls=np.vstack(models).T

    # Calculate squared differences for all grid points (vectorized)
    diff = np.sum((mdls - values) ** 2, axis=1)

    # Find the index of the nearest neighbor
    nearest_idx = np.argmin(diff)

    return nearest_idx


##################################################################

def scalar(array):
    if array.size==0:
        return -9.999999
    elif array.size==1:
        return array.item()
    else:
        return array[0].item()

##################################################################

def read_obs(filename,valid_lines):
    obsdata={}

    # read first line, used as dict keys
    with open(filename) as f:
        alllines=f.readlines()
        line=alllines[0].replace('#','').replace('# ','').replace('#\t','')

        # read keys
        #keys=re.sub('\s+',' ',line).strip().split(' ')
        keys = re.sub(r'\s+', ' ', line).strip().split(' ')

    f.close()

    # read values/columns
    with open(filename) as f:
        alllines=f.readlines()
        lines=alllines[1:]
        for i in range(len(keys)):
            #get_col = lambda col: (re.sub('\s+',' ',line).strip().split(' ')[i] for line in lines if line)
            get_col = lambda col: (re.sub(r'\s+', ' ', line).strip().split(' ')[i] for line in lines if line)

            val=np.array([float(a) for a in get_col(i)],dtype=np.float64)
            obsdata[keys[i]]=val
            keys[i] + ": "+str(val) 
    f.close()

    # convert to pandas df
    obsdata=pd.DataFrame(obsdata)

    obsdata_rows_in_file=len(obsdata.index)

    # remove rows with upper limit emissivities --> would cause trouble with calculation of
    # degrees of freedom, chi2 etc.
    mask={}
    for kk in obsdata.keys():
        if kk.upper() in valid_lines:
            line=kk.upper()
            uc='UC_'+kk.upper()
            """
            # replace with NaN
            #obsdata.loc[obsdata[line] <= obsdata[uc], line] = np.nan
            """
            mask[line]=( (obsdata[line] > obsdata[uc]) & (obsdata[line]>0) )

    finalmask=[True for x in range(len(obsdata))]
    for mm in mask.keys():
        finalmask=np.logical_and(finalmask,mask[mm])

    obsdata=obsdata[finalmask]

    obsdata_rows_selected=len(obsdata.index)

    for kk in obsdata.keys():
        obsdata[kk]=np.array(obsdata[kk])

    obsdata=obsdata.reset_index()
    del obsdata['index']

    return obsdata, obsdata_rows_in_file, obsdata_rows_selected

##################################################################

def write_result(result,outfile,domcmc):
    result=np.array(result,dtype=object)

    tmpoutfile=outfile+'.tmp'

    # extract the results
    r=result.transpose()

    if not domcmc:
        pixid,ra,de,cnt,dgf,chi2,n,T,width,str_taus,XCO,str_lines=r
        out=np.column_stack((pixid,ra,de,cnt,dgf,chi2,n,T,width,str_taus,XCO,str_lines))
        np.savetxt(tmpoutfile,out,\
            fmt="%d\t%.8f\t%.8f\t%d\t%d\t%.4f\t%.2f\t%.2f\t%.2f\t%s\t%.4f\t%s", \
            header="ID\tRA\tDEC\tcnt\tdgf\tchi2\tlog_n\tT\twidth\ttau_lines\tXCO_19\tlines_obs")
    else:
        pixid,ra,de,cnt,dgf,chi2,n,n_up,n_lo,T,T_up,T_lo,width,width_up,width_lo,str_taus,XCO,str_lines=r
        out=np.column_stack((pixid,ra,de,cnt,dgf,chi2,n,n_up,n_lo,T,T_up,T_lo,width,width_up,width_lo,str_taus,XCO,str_lines))
        np.savetxt(tmpoutfile,out,\
            fmt="%d\t%.8f\t%.8f\t%d\t%d\t%.4f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%.4f\t%s", \
            header="ID\tRA\tDEC\tcnt\tdgf\tchi2\tlog_n\te_n1\te_n2\tT\te_T1\te_T2\twidth\te_width1\te_width2\ttau_lines\tXCO_19\tlines_obs")

    # clean up
    #replacecmd="sed -e\"s/', '/|/g;s/'//g;s/\[//g;s/\]//g\""
    replacecmd = "sed -e\"s/', '/|/g;s/'//g;s/\\[//g;s/\\]//g\""
    os.system("cat "+tmpoutfile + "| "+ replacecmd + " > " + outfile)
    os.system("rm -rf "+tmpoutfile)

    return 

##################################################################

def makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile):
    try:
        fig = plt.figure(figsize=(7.5,6))
        ax = plt.gca()
        sliceindexes=np.where(this_slice==this_bestval)
        slicex=x[sliceindexes]
        slicey=y[sliceindexes]
        slicez=z[sliceindexes]
        slicex=np.array(slicex)
        slicey=np.array(slicey)
        slicez=np.array(slicez)

        if len(slicex) == 0 or len(slicey) == 0 or len(slicez) == 0:
            raise ValueError("One of the input arrays is empty.")

        # Combine data into a DataFrame
        data = pd.DataFrame({'slicex': slicex, 'slicey': slicey, 'slicez': slicez})

        # Group by unique (slicex, slicey) and aggregate slicez (e.g., mean)
        aggregated_data = data.groupby(['slicex', 'slicey'], as_index=False).mean()

        # Extract aggregated data
        slicex = aggregated_data['slicex'].values
        slicey = aggregated_data['slicey'].values
        slicez = aggregated_data['slicez'].values

        if DEBUG:
            
            from tabulate import tabulate
            print("SLICES FOR PLOTTING - slicex:")
            print(f"Range of slicex: {slicex.min()} to {slicex.max()}")
            print(tabulate(pd.DataFrame(slicex), headers='keys', tablefmt='psql'))
            print()
            print("SLICES FOR PLOTTING - slicey:")
            print(f"Range of slicey: {slicey.min()} to {slicey.max()}")
            print(tabulate(pd.DataFrame(slicey), headers='keys', tablefmt='psql'))
            print()
            print("SLICES FOR PLOTTING - slicez:")
            print(f"Range of slicez: {slicez.min()} to {slicez.max()}")
            print(tabulate(pd.DataFrame(slicez), headers='keys', tablefmt='psql'))
            print()


        if len(slicez)>3:
            # Set up a regular grid of interpolation points
            xi, yi = np.linspace(slicex.min(), slicex.max(), 60), np.linspace(slicey.min(), slicey.max(), 60)
            xi, yi = np.meshgrid(xi, yi)
            # Interpolate using Rbf
            rbf = Rbf(slicex, slicey, slicez, function='cubic')
            zi = rbf(xi, yi)

            q=[0.999]
            vmax=np.quantile(slicez,q)
            zi[zi>vmax]=vmax

            # replace nan with vmax (using workaround)
            val=-99999.9
            zi[zi==0.0]=val
            zi=np.nan_to_num(zi)
            zi[zi==0]=vmax
            zi[zi==val]=0.0

            # plot
            pl2=plt.imshow(zi, vmin=slicez.min(), vmax=slicez.max(), origin='lower', extent=[slicex.min(), slicex.max(), slicey.min(), slicey.max()],aspect='auto',cmap=cmap)
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            clb=fig.colorbar(pl2)
            clb.set_label(label=zlabel,size=16)
            clb.ax.tick_params(labelsize=18)
        #####################################

        fig.subplots_adjust(left=0.13, bottom=0.12, right=0.93, top=0.94, wspace=0, hspace=0)
        fig = gcf()

        fig.suptitle(title, fontsize=18, y=0.99)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

        fig.savefig(pngoutfile,bbox_inches='tight')
        plt.close()

        ######################################

    except Exception as e:
        if DEBUG:
            print(f"[ERROR] Type: {type(e).__name__}")
            print(f"[ERROR] Message: {e}")
        print(f"[ERROR] Creating Plot {pngoutfile} failed.")

##################################################################
##################################################################
##################################################################
################# The Dense Gas Toolbox ##########################
##################################################################
##################################################################

def dgt(obsdata_file,powerlaw,userT,userWidth,userTau,snr_line,snr_lim,plotting,domcmc,use_pt,nsteps,type_of_models,usecsv,n_cpus=1):

    interp=False    # interpolate loglike on model grid (for mcmc sampler)

    # import dgt_config.py as conf using importlib
    dgt_config_file='./models_'+type_of_models+'/dgt_config.py'
    spec = importlib.util.spec_from_file_location("conf", dgt_config_file)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    # check user inputs (T and width)
    valid_T=conf.valid_T
    valid_W=conf.valid_W
    valid_Tau=conf.valid_Tau

    if userT in valid_T and userWidth in valid_W:
        userinputOK=True
    else:
        userinputOK=False
        print("!!! User input (temperature or width) invalid. Exiting.")
        print("!!!")
        exit()

    # Valid (i.e. modeled) input molecular lines are:
    valid_lines=conf.valid_lines

    if not snr_line in valid_lines:
        print("!!! Line for SNR limit is invalid. Must be one of:")
        print(valid_lines)
        print("!!!")
        exit()

    ###########################
    ### get observations ######
    ###########################

    obs,rows_in_file,rows_selected=read_obs(obsdata_file,valid_lines)

    print("[INFO] Total rows in file: "+str(rows_in_file)+". Usable rows after removing upper limits: "+str(rows_selected))

    if DEBUG:
        from tabulate import tabulate
        print("OBS:")
        print(tabulate(pd.DataFrame(obs), headers='keys', tablefmt='psql'))
        print()
 

    ###########################
    ##### validate input ######
    ###########################

    # check for id in input file
    have_id=False

    # check for coordinates in input file
    have_radec=False
    have_ra_special=False

    if 'RA' in obs.keys() and 'DEC' in obs.keys():
        have_radec=True
    elif '#RA' in obs.keys() and 'DEC' in obs.keys():
        have_radec=True
        have_ra_special=True
    else:
        have_radec=False

    if 'id' in obs.keys():
        have_id=True
        id_pixel=obs['id']

    if 'ID' in obs.keys():
        have_id=True
        id_pixel=obs['ID']

    if have_id:
        # make sure the id is an integer
        id_pixel=np.array([int(i) for i in id_pixel])

    if not have_radec:
        print("!!!")
        print("!!! No coordinates found in input ascii file. Check column header for 'RA' and 'DEC'. Exiting.")
        print("!!!")
        exit()

        
    # count number of lines in input data
    ct_l=0
    obstrans=[]    # this list holds the user input line keys
    for key in obs.keys():
        if key in valid_lines:
            ct_l+=1
            obstrans.append(key)

    # labels
    labels = ['n','T','width']

    # Validity check of Tau
    ct_t=0
    reduce_dgf=0

    mytau=[]    # user may have entered information about optical depths of lines that are not among observed transitions
                # e.g. when tau_fiducial() is used
                # mytau holds only the relevant optical depths 
    if isinstance(userTau,list):   # optical depth is fixed for all lines
        TauIsFree=False
        for trans_tau in userTau:
            this_trans,this_tau=trans_tau.split('_')
            this_trans=str(this_trans)
            this_tau=float(this_tau)
            if this_trans in valid_lines and this_tau in valid_Tau:
                ####################################################
                # TODO: Check that Tau is same for all transitions
                # TODO: of same species (higher-J)
                ####################################################
                this_tau=float(this_tau)
                if this_trans in obstrans:
                    ct_t+=1
                    mytau.append(trans_tau)
                    labels.append('tau_'+this_trans)
            else:
                tau_error(valid_lines,valid_Tau)

    elif isinstance(userTau,int):  # optical depth is a free parameter
        TauIsFree=True

        if not userTau==0: tau_error(valid_lines,valid_Tau)
        else: mytau=0

        # Calculate reduce_dgf for this case
        ####################################################
        species=np.unique(np.array([t.replace('10','').replace('21','').replace('32','').replace('43','') for t in obstrans]))
        reduce_dgf=len(species)

    else:
        tau_error(valid_lines,valid_Tau) 

    # Only continue if:
    # i) number of molecular lines is > number of free parameters and
    # ii) in case Tau is fixed that all lines have a tau value
    if userT>0 and userWidth>0: dgf=ct_l-1-reduce_dgf
    elif userT>0 or userWidth>0: dgf=ct_l-2-reduce_dgf    # degrees of freedom = nrlines-2 if temperature is fixed. Free parameters: n,width
    else: dgf=ct_l-3-reduce_dgf                       # Free parameters: n,T,width

    print('[INFO] Degrees of Freedom for this setup/input: ',dgf)

    if not dgf>0:
        print("[ERROR] Number of observed lines too low. Degrees of Freedom <1. Try a fixed temperature or check column header. Valid lines are: ")
        print(valid_lines)
        exit()

    if not ct_t==ct_l and not TauIsFree:
        print("[ERROR] Tau value is missing for some line(s). Exiting. ")
        exit()


    if have_ra_special:
        ra=np.array(obs['#RA'])
    else:
        ra=np.array(obs['RA'])

    de=np.array(obs['DEC'])


    #############################################################################
    # Check input observations for lowest J CO line (used for normalization)
    #############################################################################
    have_co10=False
    have_co21=False
    have_co32=False
    have_co43=False

    # loop through observed lines/transitions
    for t in obstrans:
        if t=='CO10': have_co10=True
        if t=='CO21': have_co21=True
        if t=='CO32': have_co32=True
        if t=='CO43': have_co43=True

    if have_co10: normtrans='CO10'; uc_normtrans='UC_CO10'
    elif have_co21: normtrans='CO21'; uc_normtrans='UC_CO21'
    elif have_co32: normtrans='CO32'; uc_normtrans='UC_CO32'
    elif have_co43: normtrans='CO43'; uc_normtrans='UC_CO43'
    else:
        print("[ERROR] No CO line found in input data file. Check column headers for 'CO10', 'CO21', 'CO32' or 'CO43'. Exiting.")
        exit()



    ###########################
    ##### get the models ######
    ###########################
    mdl={}
    mdl = read_grid_ndist(obstrans,userT,userWidth,mytau,powerlaw,type_of_models,usecsv)
    print("[INFO] Grid size: "+str(len(mdl['tkin'])))

    if DEBUG:
        from tabulate import tabulate
        print("MODELS ORIG:")
        print(tabulate(pd.DataFrame(mdl), headers='keys', tablefmt='psql'))
        print()


    #############################################################################
    #############################################################################
    # Calculate line ratios and save in new dictionary
    # use line ratios (normalize to lowest CO transition in array) to determine chi2
    # note that the abundances are fixed by design of the model grid files
    #############################################################################
    #############################################################################
    lr={}

    # loop through observed lines/transitions
    print("[INFO] Calculating line ratios")
    for t in obstrans:
        if t!=normtrans:
            tm=linename_obs2mdl(t)
            normtransm=linename_obs2mdl(normtrans)
            # calc line ratios
            lr[t]=obs[t]/obs[normtrans]
            print(tm+'/'+normtrans)
            mdl[t]=mdl[tm].to_numpy()/mdl[normtransm].to_numpy()
            del mdl[tm]

            uc='UC_'+t
            lr[uc]=abs(obs[uc]/obs[t]) + abs(obs[uc_normtrans]/obs[normtrans])

    if DEBUG:
        print("MODELS Line Ratios:")
        print(tabulate(pd.DataFrame(mdl), headers='keys', tablefmt='psql'))
        print()


    #############################################################
    #############################################################
    # loop through pixels, i.e. rows in ascii input file
    #############################################################
    #############################################################
    result=[]

    if not have_id:
        pixels=range(len(ra))
    else:
        pixels=id_pixel

    for p,pixnr in enumerate(pixels):

        if not have_id:
            pixnr=p+1

        #################################
        ####### calculate chi2 ##########
        #################################
        diff={}
        for t in obstrans:
            if t!=normtrans:
                uc='UC_'+t
                if obs[t][p]>obs[uc][p] and obs[t][p]>0.0:
                    diff[t]=np.array(((lr[t][p]-mdl[t])/lr[uc][p])**2)
                else:
                    diff[t]=np.nan*np.zeros_like(mdl[t])

        if DEBUG:
            print("DIFF Pixel "+str(p))
            print(tabulate(pd.DataFrame(diff), headers='keys', tablefmt='psql'))
            print()


        # vertical stack of diff arrays
        vstack=np.vstack(list(diff.values()))
        # sum up diff of all line ratios--> chi2
        chi2=vstack.sum(axis=0)

        if DEBUG:
            print("CHI2 Pixel "+str(p))
            print(tabulate(pd.DataFrame(chi2), headers='keys', tablefmt='psql'))
            print()


        # if model correct, we expect:
        # nu^2 ~ nu +/- sqrt(2*nu)

        # make a SNR cut using line and limit from user
        uc='UC_'+snr_line
        SNR=round(obs[snr_line][p]/obs[uc][p],2)

        width=ma.array(mdl['width'])
        densefrac=ma.array(mdl['fdense_thresh'])

        # filter out large values (since loglike is propto 10**chi2 --> chi2>100 leads to crazy high numbers 
        chi2lowlim,chi2uplim=0,9999
        #chi2lowlim,chi2uplim=np.quantile(chi2,[0.0,0.95])

        # create masks
        # invalid (nan) values of chi2
        chi2=ma.masked_invalid(chi2)
        mchi2invalid=ma.getmask(chi2)

        # based on chi2
        chi2=ma.array(chi2)
        chi2=ma.masked_outside(chi2, chi2lowlim, chi2uplim)
        mchi2=ma.getmask(chi2)
        # based on densefrac
        densefraclowlim=0.
        densefracuplim=99999. 
        densefrac=ma.masked_outside(densefrac,densefraclowlim,densefracuplim)
        mwidth=ma.getmask(densefrac)
 
        # combine masks
        m1=ma.mask_or(mchi2,mwidth)
        m=ma.mask_or(m1,mchi2invalid)

        #############
        # APPLY MASKS
        #############
        width=ma.array(width,mask=m) 
        densefrac=ma.array(densefrac,mask=m)
        chi2=ma.array(chi2,mask=m)

        # n,T
        grid_n=mdl['n_mean_mass']
        n=ma.array(grid_n,mask=m)

        grid_T=mdl['tkin']
        T=ma.array(grid_T,mask=m)

        # line optical depths (tau), for (1-0) transition
        grid_tau={}
        tau={}
        for kk in mdl.keys():
            if kk[0:3]=='tau':
                grid_tau[kk]=mdl[kk]
                tau[kk]=ma.array(grid_tau[kk],mask=m)


        ###########################################################
        ########## find best fit set of parameters ################
        ############ from chi2 credible interval ##################
        ###########################################################

        # These limits correspond to +/-1 sigma error
        if dgf>0:
            cutoff=0.05  # area to the right of critical value; here 5% --> 95% confidence  --> +/- 2sigma
            #cutoff=0.32  # area to the right of critical value; here 32% --> 68% confidence --> +/- 1sigma
            deltachi2=scipychi2.ppf(1-cutoff, dgf)
        else:
            print("DGF is zero or negative.")

        # The minimum
        # find best fit set of parameters 
        chi2min=np.ma.min(chi2)
        bestfitindex=ma.where(chi2==chi2min)[0]
        bestchi2=scalar(chi2[bestfitindex].data)
        bestn=scalar(n[bestfitindex].data)
        bestwidth=scalar(width[bestfitindex].data)
        bestT=scalar(T[bestfitindex].data)
        bestdensefrac=scalar(densefrac[bestfitindex].data)
        bestTau=''
        for kk in tau.keys():
            this_tau=tau[kk]
            this_bestTau=scalar(this_tau[bestfitindex].data)
            bestTau+=';'+str(this_bestTau)
        bestTau=bestTau[1:]
        bestchi2=round(bestchi2,2)
        bestreducedchi2=round(bestchi2/dgf,2)


        print('[INFO] Minimum reduced chi2 is '+str(bestreducedchi2))
        print('[INFO] Log density at chi2 minimum is '+str(round(bestn,2)))


        #################################################
        ########## Show Chi2 result on screen ###########
        #################################################

        if not domcmc:

            ############### LOOKUP ##################
            ######### CONVERSION FACTOR XCO #########
            ICO_obs=obs[normtrans]

            if TauIsFree:

                tauarr=[]
                tauval=[]

                for ii,lbl in enumerate(labels[3:]):
                    this_species=lbl.split('_')[1]
                    bestTau_val=bestTau.split(';')[ii]
                    tauarr.append(mdl['tau_'+this_species])
                    tauval.append(float(bestTau_val))

                ICO_mdl_index_nTW = find_nearest( [ mdl['n_mean_mass'], mdl['tkin'], mdl['width'] ]+tauarr , [ float(bestn), float(bestT) , float(bestwidth) ]+tauval )

            else:
                ICO_mdl_index_nTW = find_nearest( [ mdl['n_mean_mass'], mdl['tkin'], mdl['width'] ] , [ float(bestn), float(bestT) , float(bestwidth) ] )

            final_mask = [False for x in range(len(mdl['tkin']))]
            final_mask[ICO_mdl_index_nTW]=True

            XCO_rows=mdl[final_mask].reset_index()
            COLDENS=XCO_rows[normtransm]
            XCO=1./COLDENS
            XCO=round(float(XCO.values[0]/1e19),4)

            if SNR>snr_lim and bestn>0:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("#### Bestfit Parameters for pixel nr. "+str(pixnr)+" ("+str(round(ra[p],5))+","+str(round(de[p],5))+ ") ####")
                print("chi2\t\t" + str(bestchi2))
                print("red. chi2\t\t" + str(bestreducedchi2))
                print("log n\t\t" + str(bestn))
                print("T\t\t" + str(bestT))
                print("Taus\t\t" + str(bestTau))
                print("Width\t\t" + str(bestwidth))
                print("XCO_19\t\t" + str(XCO))
                print()

                #############################################
                # save results in array for later file export
                result.append([pixnr,ra[p],de[p],ct_l,dgf,bestchi2,bestn,bestT,bestwidth,bestTau,XCO,obstrans])
                do_this_plot=True
            else:
                print("!-!-!-!-!-!")
                print("Pixel no. " +str(pixnr)+ " --> SNR too low or density<0.")
                print()
                result.append([pixnr,ra[p],de[p],ct_l,dgf,-99999.9,-99999.9,-99999.9,-99999.9,'-99999.9',-99999.9,obstrans])
                do_this_plot=False

        ###################################################################
        ###################################################################
        ################################# MCMC ############################
        ###################################################################

        if domcmc:
            if SNR>snr_lim and bestn>0:
                print("[INFO] Preparing MCMC for pixel "+str(pixnr))
 
                #### Create directory for output png files ###
                if not os.path.exists('./results2/'):
                    os.makedirs('./results2/')

                if TauIsFree:
                    ndim, nwalkers = 3+len(tau), 3*(3+len(tau))
                else:
                    ndim, nwalkers = 3, 3*(3+len(tau))

                # model grid in results file
                grid_theta=[n,T,width]

                labels=['log n','T','width']
                if TauIsFree:
                    for ii,ss in enumerate(tau.keys()):
                        grid_theta = grid_theta + [tau[ss]]
                        labels.append(ss)
                
                grid_theta = np.array(grid_theta,dtype=np.float64)
                grid_loglike  = -0.5 * chi2

                if DEBUG:
                    print("LOGLIKE")
                    print(tabulate(pd.DataFrame(grid_loglike), headers='keys', tablefmt='psql'))
                    print()

                    """
                    print("GRID_THETA")
                    print(tabulate(pd.DataFrame(grid_theta), headers='keys', tablefmt='psql'))
                    print()
                    """

                # parameters for initial "burn-in" sampling to avoid stuck walkers
                nreps = 3
                nsims_burnin = 200
                                
                # set up backend (from dgt v1.7)
                status_filename = './results2/'+obsdata_file[:-4]+'_mcmc_'+str(p+1)+'.h5'
                backend = emcee.backends.HDFBackend(status_filename)
                # reset if already exists
                #backend.reset(nwalkers, ndim)

                if not os.path.exists(status_filename):
                   # perform sampling
                    mcmc.mymcmc(grid_theta, grid_loglike, ndim, nwalkers, interp, nsteps, labels, conf, nreps, nsims_burnin, backend, n_cpus=n_cpus, pixelnr=str(pixnr),do_ptmcmc=use_pt)
                else:
                    print('[INFO] Re-using previously generated chain for pixel id'+str(pixnr))


                ########## MAKE CORNER PLOT #########
                if not use_pt:
                    outpngfile="./results2/"+obsdata_file[:-4]+"_mcmc_"+str(pixnr)+".png"
                    bestn_mcmc_val,bestn_mcmc_upper,bestn_mcmc_lower,\
                        bestT_mcmc_val,bestT_mcmc_upper,bestT_mcmc_lower,\
                        bestW_mcmc_val,bestW_mcmc_upper,bestW_mcmc_lower,\
                        taulist = \
                        mcmc_corner_plot(status_filename,outpngfile,labels,ndim,pixelnr=str(pixnr))
                else:
                    outpngfile="./results2/"+obsdata_file[:-4]+"_mcmc_"+str(pixnr)+".png"
                    bestn_mcmc_val,bestn_mcmc_upper,bestn_mcmc_lower,\
                        bestT_mcmc_val,bestT_mcmc_upper,bestT_mcmc_lower,\
                        bestW_mcmc_val,bestW_mcmc_upper,bestW_mcmc_lower,\
                        taulist = \
                        mcmc_corner_plot_ptmcmc(status_filename,outpngfile,labels,ndim,pixelnr=str(pixnr))
    


                ############### LOOKUP ##################
                ######### CONVERSION FACTOR XCO #########
                ICO_obs=obs[normtrans]


                if float(bestn_mcmc_val)>0.0 and float(bestT_mcmc_val)>0.0 and float(bestW_mcmc_val)>0.0:
                    if TauIsFree:
                        
                        tauarr=[]
                        tauval=[]
                        for ii,lbl in enumerate(labels[3:]):
                            this_species=lbl.split('_')[1]
                            _,bestTau_mcmc_val,bestTau_mcmc_upper,bestTau_mcmc_lower=taulist[ii]
                            tauarr.append(mdl['tau_'+this_species])
                            tauval.append(float(bestTau_mcmc_val))
    
                        ICO_mdl_index_nTW = find_nearest( [ mdl['n_mean_mass'], mdl['tkin'], mdl['width'] ]+tauarr , [ float(bestn_mcmc_val), float(bestT_mcmc_val) , float(bestW_mcmc_val) ]+tauval )
     
                    else:
                        ICO_mdl_index_nTW = find_nearest( [ mdl['n_mean_mass'], mdl['tkin'], mdl['width'] ] , [ float(bestn_mcmc_val), float(bestT_mcmc_val) , float(bestW_mcmc_val) ] )
        
                    final_mask = [False for x in range(len(mdl['tkin']))]
                    final_mask[ICO_mdl_index_nTW]=True
    
                    XCO_rows=mdl[final_mask].reset_index()
                    COLDENS=XCO_rows[normtransm]
                    XCO=1./COLDENS
                    XCO=round(float(XCO.values[0]/1e19),4)
                else:
                    XCO=np.nan

                #########################################
                #########################################

                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("#### Bestfit Parameters for pixel nr. "+str(pixnr)+" ("+str(round(ra[p],5))+","+str(round(de[p],5))+ ") ####")
                print("log n\t\t" + str(bestn_mcmc_val) + " " + str(bestn_mcmc_upper) + " " + str(bestn_mcmc_lower))
                print("T\t\t" + str(bestT_mcmc_val) + " " + str(bestT_mcmc_upper) + " " + str(bestT_mcmc_lower))
                print("Width\t\t" + str(bestW_mcmc_val) + " " + str(bestW_mcmc_upper) + " " + str(bestW_mcmc_lower))
                if len(labels)>3:
                    for ii,lbl in enumerate(labels[3:]):
                        _,bestTau_mcmc_val,bestTau_mcmc_upper,bestTau_mcmc_lower=taulist[ii]
                        print(lbl+"\t\t" + str(bestTau_mcmc_val) + " " + str(bestTau_mcmc_upper) + " " + str(bestTau_mcmc_lower))
                print("XCO_19\t\t" + str(XCO))
                print()

                #############################################
                # save results in array for later file export
                result.append([pixnr,ra[p],de[p],ct_l,dgf,float(bestchi2),float(bestn_mcmc_val),float(bestn_mcmc_upper),float(bestn_mcmc_lower),\
                                                    float(bestT_mcmc_val),float(bestT_mcmc_upper),float(bestT_mcmc_lower),\
                                                    float(bestW_mcmc_val),float(bestW_mcmc_upper),float(bestW_mcmc_lower),\
                                                    str(taulist).replace(', ',';'),\
                                                    XCO,\
                                                    obstrans])
                do_this_plot=True 
                ###################################################################
                ###################################################################

            elif SNR<snr_lim:
                print("[INFO] Skipping pixel id "+str(pixnr)+", because SNR for line " +str(snr_line) + " is lower than user constraint.")
                do_this_plot=False

            elif bestn<=0:
                print("[INFO] Skipping pixel id "+str(pixnr)+", because log-likelihood could not be constrained from chi2.")
                do_this_plot=False


        ############################################
        ################ Make Figures ##############
        ############################################

        # Plotting
        if SNR>snr_lim and plotting==True and bestn>0 and do_this_plot:

            #### Create directory for output png files ###
            if not os.path.exists('./results2/'):
                os.makedirs('./results2/')

            # zoom-in variables
            idx=np.where(chi2<bestchi2+deltachi2)
            zoom_n=n[idx].compressed()
            zoom_chi2=chi2[idx].compressed()
            zoom_width=width[idx].compressed()


            ########################## PLOT 1 #############################

            # combine 4 plots to a single file
            fig, ax = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(11.5,8))
            # Chi2 vs n plot
            # Note: n is log already!
            ax[0,0].scatter(chi2, n, c=width, cmap='Accent',marker=',',s=4,vmin=width.min(),vmax=width.max())
            ax[0, 0].set_ylabel('$\\log\\ n$')

            pl1=ax[0,1].scatter(zoom_chi2, zoom_n, c=zoom_width, cmap='Accent',marker=',',s=9,vmin=width.min(),vmax=width.max())
            #fig.colorbar(pl1,ax=ax[0,1],label='$\mathsf{width}$')
            fig.colorbar(pl1, ax=ax[0, 1], label='$\\mathsf{width}$')

            # Chi2 vs T plot
            ax[1,0].scatter(chi2, np.log10(T),c=width, cmap='Accent',marker=',',s=4,vmin=width.min(),vmax=width.max())
            ax[1, 0].set_xlabel('$\\chi^2$')
            ax[1, 0].set_ylabel('$\\log\\ T$')

            # Chi2 vs T plot zoom-in
            zoom_T=T[chi2<bestchi2+deltachi2].compressed()
            pl2=ax[1,1].scatter(zoom_chi2, np.log10(zoom_T),c=zoom_width, cmap='Accent',marker=',',s=9,vmin=width.min(),vmax=width.max())
            ax[1, 1].set_xlabel('$\\chi^2$')
            fig.colorbar(pl2, ax=ax[1, 1], label='$\\mathsf{width}$')
 
            # plot
            fig.subplots_adjust(left=0.06, bottom=0.06, right=1, top=0.96, wspace=0.04, hspace=0.04)
            fig = gcf()
            fig.suptitle('Pixel: ('+str(p)+') SNR('+snr_line+'): '+str(SNR), fontsize=14, y=0.99) 
            chi2_filename=obsdata_file[:-4]+"_"+str(pixnr)+'_chi2.png'
            fig.savefig('./results2/'+chi2_filename) 
            #plt.show()
            plt.close()


            ########################## PLOT 2 #############################
            # all parameters free: (n,T) vs. chi2
            if userT==0 and userWidth==0:

                  x=zoom_n  # n is log already
                  y=np.log10(zoom_T)
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_width
                  this_bestval=bestwidth
                  xlabel = '$\\log\\ n\\ [cm^{-3}]$'
                  ylabel = '$\\log\\ T\\ [K]$'
                  zlabel = '$\\log\\ \\chi^2$'

                  title='Pixel: '+str(pixnr)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results2/'+obsdata_file[:-4]+"_"+str(pixnr)+'_nT.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)

                  ########################## PLOT 3 #############################
                  # all parameters free: (n,width) vs. chi2
                  x=zoom_n  # n is log already
                  y=zoom_width
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_T
                  this_bestval=bestT
                  xlabel = '$\\log\\ n\\ [cm^{-3}]$'
                  ylabel = '$width\\ [dex]$'
                  zlabel = '$\\log\\ \\chi^2$'

                  title='Pixel: '+str(pixnr)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results2/'+obsdata_file[:-4]+"_"+str(pixnr)+'_nW.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)
 
            # width fixed: (n,T) vs. chi2
            elif userT==0 and userWidth>0:
                  x=zoom_n # n is log already
                  y=np.log10(zoom_T)
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_width
                  this_bestval=bestwidth
                  xlabel = '$\\log\\ n\\ [cm^{-3}]$'
                  ylabel = '$\\log\\ T\\ [K]$'
                  zlabel = '$\\log\\ \\chi^2$'

                  title='Pixel: '+str(pixnr)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results2/'+obsdata_file[:-4]+"_"+str(pixnr)+'_nT_fixedW.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)

            # T fixed: (n,width) vs. chi2
            elif userT>0 and userWidth==0:
                  x=zoom_n  # n is log already
                  y=zoom_width
                  z=np.log10(zoom_chi2)
                  this_slice=zoom_T
                  this_bestval=bestT
                  xlabel = '$\\log\\ \\n\\ [cm^{-3}]$'
                  ylabel='$width\\ [dex]$'
                  zlabel = '$\\log\\ \\chi^2$'

                  title='Pixel: '+str(pixnr)+ ' | SNR('+snr_line+')='+str(SNR)
                  pngoutfile='results2/'+obsdata_file[:-4]+"_"+str(pixnr)+'_nW_fixedT.png'

                  makeplot(x,y,z,this_slice,this_bestval,xlabel,ylabel,zlabel,title,pngoutfile)


        del diff,chi2,n,T,width,densefrac,mchi2,mchi2invalid,mwidth,m1,m,grid_n,grid_T

    ################################################
    ################################################
    # write result to a new output table
    if not domcmc:
        outtable=obsdata_file[:-4]+"_nT.txt"
    else:
        outtable=obsdata_file[:-4]+"_nT_mcmc.txt"
    resultfile="./results2/"+outtable
    write_result(result,resultfile,domcmc)


    #plt.show(block=True)

##################################################################################
##################################################################################

def tau_error(valid_lines,valid_taus):
    print("!!!")
    print("!!! Make sure that tau is either 0 (tau as free parameter) or enter tau for each LOWEST-J line manually in the format:")
    print("!!! e.g. ['CO10_5.0','13CO10_0.1'] using a combination of valid lines and valid optical depths.")
    print("!!! Valid lines are: ",valid_lines)
    print("!!!")
    print("!!! Valid Taus are: ",valid_taus)
    print("!!!")
 
    exit()

##################################################################################
##################################################################################

def tau_fiducial():

    return ['CO10_6.5','CO21_6.5','CO32_6.5',\
            'HCN10_0.8','HCN21_0.8','HCN32_0.8',\
            'HCOP10_1.5','HCOP21_1.5','HCOP32_1.5',\
            '13CO10_0.2','13CO21_0.2','13CO32_0.2',\
            'C18O10_0.1','C18O21_0.1','C18O32_0.1']
#            'HNC10_0.8','HNC21_0.8','HNC32_0.8',\ 
#            'C17O10_0.1','C17O21_0.1','C17O32_0.1',\
#            'CS10_0.8','CS21_0.8','CS32_0.8']

