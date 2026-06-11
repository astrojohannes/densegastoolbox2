#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import os
import psutil
import requests
import glob

try:
    import dask.dataframe as dd
    from dask import delayed
except ImportError:  # Dask is useful for large grids but not required for CSV/small-grid use.
    dd = None
    delayed = None

############################################################

# Function to check free space on the disk
def check_free_space(path='.'):
    # Get disk usage statistics for the filesystem that will hold the model file.
    disk_usage = psutil.disk_usage(path)
    # Convert free space from bytes to GB
    free_space_gb = disk_usage.free / (1024 ** 3)
    return free_space_gb

###########################################################

# Function to download a file
def download_file(url, local_path, model_size_gb):
    local_path = Path(local_path)
    # Check if the file already exists
    if local_path.exists():
        print(f"File {local_path} already exists. Skipping download.")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    free_space_gb = check_free_space(local_path.parent if local_path.parent != Path('') else '.')

    if free_space_gb < model_size_gb and "emissivities" in str(local_path):
        raise RuntimeError(
            f"Not enough disk space. Available: {free_space_gb:.2f} GB, "
            f"required: {model_size_gb} GB"
        )

    tmp_path = local_path.with_suffix(local_path.suffix + '.download')

    print(f"Downloading {url} to {local_path}...")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with tmp_path.open('wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    tmp_path.replace(local_path)
    print(f"Downloaded {local_path}")

def ensure_directory_exists(directory_path):
    directory_path = Path(directory_path)
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory {directory_path} created.")

############################################################

def read_and_save_chunks(pklfile, chunk_size=10000, chunk_prefix='chunk_', tmp_dir='tmp'):
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    chunk_index = 0
    with open(pklfile, 'rb') as pk:
        while True:
            chunk = []
            try:
                for _ in range(chunk_size):
                    a = pd.read_pickle(pk)
                    chunk.append(a)
                    n += 1
                    print(f"[INFO] Reading model groups (n,T,width fixed per group) {n}", end='\r')
            except EOFError:
                pass

            if chunk:
                chunk_df = pd.concat(chunk, ignore_index=True)
                chunk_filename = tmp_dir / f'{chunk_prefix}{chunk_index}.pkl'
                chunk_df.to_pickle(chunk_filename)
                chunk_index += 1

            if len(chunk) < chunk_size:
                print()
                break

def load_pickle(file):
    return pd.read_pickle(file)

def concatenate_chunks_dask(chunk_prefix='chunk_', tmp_dir='tmp'):
    chunk_files = sorted(glob.glob(str(Path(tmp_dir) / f'{chunk_prefix}*.pkl')))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {tmp_dir!r} with prefix {chunk_prefix!r}")

    if dd is None or delayed is None:
        # Fallback for installations without dask. This may require more RAM,
        # but avoids import-time crashes when dask is not installed.
        return pd.concat((load_pickle(file) for file in chunk_files), ignore_index=True)

    delayed_dfs = [delayed(load_pickle)(file) for file in chunk_files]
    ddf = dd.from_delayed(delayed_dfs)
    result = ddf.compute()
    return result

def read_stream_old(pklfile):
    objs = []
    n = 0
    with open(pklfile, 'rb') as pk:
        while True:
            try:
                a = pd.read_pickle(pk)
                objs.append(a)
                n += 1
                print("[INFO] Reading model groups (n,T,width fixed per group) " + str(n), end='\r')
            except EOFError:
                print()
                break

    return objs

############################################################

def read_grid_ndist(transition,usertkin,userwidth,usertau,powerlaw,type_of_models="std",usecsv=False):

    if not powerlaw:
        if usecsv: lratfile='emissivities.csv'
        else: lratfile='emissivities.pkl'
    else:
        if usecsv: lratfile='emissivities_powerlaw.csv'
        else: lratfile='emissivities_powerlaw.pkl'

    gridfile='models_'+type_of_models+'/'+lratfile
    ensure_directory_exists('./models_'+type_of_models+'/')

    if type_of_models=="std43_incl_HCN_excl_C18O":
        model_size_gb = 40
    elif type_of_models=="co":
        model_size_gb = 9.2
    elif type_of_models=="thick":
        model_size_gb = 12
    else:
        model_size_gb = 33  # std

    # Define URLs and local paths for the files
    urls = [
            "https://www.jpuschnig.com/dgt/"+gridfile
    ]
    local_paths = [
            gridfile
    ]
    # Download model files if needed
    for url, local_path in zip(urls, local_paths):
        download_file(url, local_path, model_size_gb)

    if usecsv:
        print("[INFO] Reading models from csv")
        grid=pd.read_csv(gridfile)
    else:
        """
        grids = read_stream(gridfile)
        print("[INFO] Concat model groups")
        # concat to single df
        grid=pd.concat(grids).reset_index()
        """
        read_and_save_chunks(gridfile)
        print("[INFO] Concat model groups")
        # concat to single df
        grid = concatenate_chunks_dask()

    #if not powerlaw: grid.to_csv("models/emissivities.csv",index=False)
    #else: grid.to_csv("models/emissivities_powerlaw.csv",index=False)
    #exit()

    print("[INFO] Down-select models to user input")

    print("Lines")
    # limit to user lines
    userlines=[linename_obs2mdl(x) for x in transition]
    userspecies=[linename_obs2mdl(x).replace('10','').replace('21','').replace('32','').replace('43','') for x in userlines]
    mdlcols=['n_mean','n_mean_mass','tkin','width','fdense_thresh','fdense_pl','pl']
    keepcols=userlines+mdlcols

    for kk in list(grid.keys()):
        if kk[0:3]!='tau' and kk not in keepcols:
            #print("Removing column "+str(kk))
            del grid[kk]
        elif kk[0:3]=='tau':
            this_species=kk.split('_')[1]
            if this_species not in userspecies:
                #print("Removing column "+str(kk))
                del grid[kk]

    grid = grid.drop_duplicates()

    print("Tkin")
    # limit to values at user temperature
    if usertkin>0:
        mask_T=(grid['tkin'] == usertkin)
    else:
        mask_T=[True for x in grid['tkin']]

    print("width")
    # limit to values at user width
    if userwidth>0:
        mask_w=(grid['width'] == userwidth)
    else:
        mask_w=[True for x in grid['width']]

    print("Tau")
    # limit to values at user tau
    if isinstance(usertau,list) and len(usertau)>0:
        # Tau is fixed
        taumask={}
        for trans_tau in usertau:
            this_trans,this_tau=trans_tau.split('_')
            this_species=linename_obs2mdl(this_trans).replace('10','').replace('21','').replace('32','').replace('43','')
            taumask[this_trans]=(grid['tau_'+this_species] == float(this_tau))

        mask_tau=taumask[list(taumask.keys())[0]]
        for tt in list(taumask.keys())[1:]:
            mask_tau=np.logical_and(mask_tau,taumask[tt])

    elif isinstance(usertau,int) and usertau==0:
        # Tau is a free parameter.
        mask_tau=[True for x in grid['tkin']]
    else:
        mask_tau=[True for x in grid['tkin']]

    # combine masks and apply
    mask=np.logical_and(mask_w,np.logical_and(mask_T,mask_tau))

    g=grid[mask].reset_index(drop=True)

    g['n_mean']=np.log10(g['n_mean'])
    g['n_mean_mass']=np.log10(g['n_mean_mass'])

    # return
    return g

###########################################################################
###########################################################################

def linename_obs2mdl(line:str):

    ll=line.lower()
    if ll=='co10': ll='12co10'
    elif ll=='co21': ll='12co21'
    elif ll=='co32': ll='12co32'
    elif ll=='co43': ll='12co43'

    return str(ll)

###########################################################################
###########################################################################

def linename_mdl2obs(line:str):

    ll=line.upper()
    if ll=='12CO10': ll='CO10'
    elif ll=='12CO21': ll='CO21'
    elif ll=='12CO32': ll='CO32'
    elif ll=='12CO43': ll='CO43'

    return str(ll)


