#!/usr/bin/python

from dgt2 import dgt

if __name__ == "__main__":

    ##############################################################
    ######## Check out the Model Parameter Space explorer ########
    ########    http://www.densegastoolbox.com/explorer   ########
    ##############################################################

    ##############################################################
    #################### USER INPUT BELOW ########################
    ##############################################################
    obsdata_file = 'ascii_galaxy.txt'    # table of observed intensities in Tmb [K km/s]
    
    ###################################
    # Note that the input file (obsdata_file) must have a 1-line
    # header, indicating the line intensities (in K km/s) via the
    # following column names:
    # 
    # CO10      CO21        CO32        CO43
    # HCN10     HCN21       HCN32       HCN43
    # HCOP10    HCOP21      HCOP32      HCOP43
    # HNC10     HNC21       HNC32       HNC43
    # 13CO10    13CO21      13CO32      13CO43
    # C18O10    C18O21      C18O32      C18O43
    # C17O10    C17O21      C17O32      C17O43
    # CS10      CS21        CS32        CS43
    #
    # NOTE: README_models provides information about the transitions available in
    # differnet model grids the user can choose from
    #
    # The uncertainties are similar, but starting with "UC_",
    # e.g. UC_CO10 or UC_HCN21
    #
    # Fiducial Tau values (based on EMPIRE) as they were used in DGT v1.X
    # 12CO:       6.5
    # 13CO:       0.2
    # C17O,C18O:  0.1
    # HCN,HNC,CS: 0.8
    # HCOP:       1.5
    #
    # #####################################################################################################
    # ###################################### User INPUT Parameters ########################################
    # #####################################################################################################

    powerlaw=True                           # logNorm or logNorm+PL density distribution
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    T=0                                     # gas temperature; use T=0 to leave as free parameter
                                            # must be one of: 10,15,20,25,30
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    W=0                                     # with of density distribution in dex; use W=0 to leave as free parameter
                                            # must be one of: 0.2,0.4,0.6,0.8
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                            # Models will be downloaded upon usage (note large file size)
                                            # see README_models.md
                                            #
    type_of_models='co'                     # which models to use (one of: std, std43, std43_incl_HCN_excl_C18O, co, coarse)
                                            # std (up to 3-2): 2 x 35GB
                                            # std43 (up to 4-3): 2 x 38GB
                                            # std43_incl_HCN_excl_C18O (up to 4-3): 2 x 38GB
                                            # thick (up to 4-3): 2 x 10GB
                                            # co (up to 3-2): 2x 10GB
                                            # coarse: currently not available
                                            #
    models_from_csv=False                   # in case of problems with unpickling the models, a csv
                                            # version of the models may be used
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tau=0                                   # line optical depths (for lowest-J transition) of given species
                                            # use tau=0 to leave as free parameter
                                            # or set for all lines using the following syntax:
                                            # ['CO10_6.5','CO21_6.5','CO32_6.5','13CO10_0.2',...]
                                            # to a fixed optical depth. Only the following tau values are allowed:
                                            #
                                            # std: [0.1,0.2,0.3] for 13CO and C18O (up to 3-2)
                                            #      [0.8,1.1,1.5] for HCN and HCOP (up to 3-2)
                                            #      [5.0,6.5,8.0] for 12CO (up to 3-2)
                                            #
                                            # std43: same as std but up to (4-3)
                                            #
                                            # std43_incl_HCN_excl_C18O:
                                            #       [0.1,0.2,0.3] for 13CO (up to 4-3)
                                            #       [0.8,1.1,1.5] for HCN, HCOP and HNC (up to 4-3)
                                            #       [5.0,6.5,8.0] for 12CO (up to 4-3)
                                            #
                                            # thick: [0.8,1.1,1.5] for HCN, HCOP and HNC (up to 4-3)
                                            #        [5.0,6.5,8.0] for 12CO (up to 4-3)
                                            #
                                            # co: [0.1,0.2,0.3] for 13CO and C18O (up to 3-2)
                                            #     [5.0,6.5,8.0] for 12CO (up to 3-2)
                                            #
                                            # or set tau='tau_fiducial' to use EMPIRE-based fixed optical depths
                                            # this reproduces results from previous old DGT v1.X
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    snr_line='CO10'                         # only use data above SNR cut in given line, should be faintest line
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    snr_lim=0                               # this is the corresponding SNR cut
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    plotting=True                           # create plots
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    domcmc=True                             # use MCMC for parameter estimation; this is recommended, but may take very long
    use_pt=False                            # if True, the PTMCMC Sampler is used instead of emcee
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    nsims=2000                              # number of MCMC simulations to perform (should be >1000, better even more)
    n_cpus = 12                             # Upper limit for number of cpus used for MCMC 
    #######################################################################################################

    # call Dense GasTool box
    dgt(obsdata_file,powerlaw,T,W,tau,snr_line,snr_lim,plotting,domcmc,use_pt,nsims,type_of_models,models_from_csv,n_cpus)

    # exit
    exit(0)
