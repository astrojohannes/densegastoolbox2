-------------------------------------------------------------------------------------
---------------------------- Standard models (32GB) ---------------------------------
-------------------------- min. RAM requirement: 64GB -------------------------------
-------------------------------------------------------------------------------------
shortname: std

The v2 standard models contain molecular line emissivities
for the following transitions and parameter space:

n=10.**(1.8+np.arange(33)*0.1) = [6.30957344e+01, 7.94328235e+01, 1.00000000e+02, 1.25892541e+02,
       1.58489319e+02, 1.99526231e+02, 2.51188643e+02, 3.16227766e+02,
       3.98107171e+02, 5.01187234e+02, 6.30957344e+02, 7.94328235e+02,
       1.00000000e+03, 1.25892541e+03, 1.58489319e+03, 1.99526231e+03,
       2.51188643e+03, 3.16227766e+03, 3.98107171e+03, 5.01187234e+03,
       6.30957344e+03, 7.94328235e+03, 1.00000000e+04, 1.25892541e+04,
       1.58489319e+04, 1.99526231e+04, 2.51188643e+04, 3.16227766e+04,
       3.98107171e+04, 5.01187234e+04, 6.30957344e+04, 7.94328235e+04,
       1.00000000e+05]
T=[0,10,15,20,25,30]
W=[0,0.2,0.4,0.6,0.8]
Tau=[0.1,0.2,0.3,
     0.8,1.1,1.5,
     5.0,6.5,8.0]

--> combinations of n,T,W: 33*5*4
--> combinations of Lines/Tau: 9^5

valid_lines=['CO10','CO21','CO32',\
        'HCN10','HCN21','HCN32',\
        'HCOP10','HCOP21','HCOP32',\
        '13CO10','13CO21','13CO32',\
        'C18O10','C18O21','C18O32']

-------------------------------------------------------------------------------------
------------------ Standard models incl. 4-3 transitions (??GB) ---------------------
-------------------------- min. RAM requirement: ??GB -------------------------------
-------------------------------------------------------------------------------------
shortname: std43

***********************
* NOT YET IMPLEMENTED *
***********************

The v2 standard models incl. (4-3) transitions contain molecular line emissivities
for the following transitions and parameter space:

n, T, W, Tau...same as standard model

--> combinations of n,T,W: 33*5*4
--> combinations of Lines/Tau: 9^5

valid_lines=['CO10','CO21','CO32','CO43',\
        'HCN10','HCN21','HCN32','HCN43',\
        'HCOP10','HCOP21','HCOP32','HCOP43',\
        '13CO10','13CO21','13CO32','13CO43',\
        'C18O10','C18O21','C18O32','C18O43']

-------------------------------------------------------------------------------------
-------------------- Thick models incl. 4-3 transitions (10GB) ----------------------
-------------------------- min. RAM requirement: 20GB -------------------------------
-------------------------------------------------------------------------------------
shortname: thick

The v2 thick models contain molecular line emissivities
for the following transitions and parameter space:

n, T, W...same as standard model
Tau=[0.8,1.1,1.5,
     5.0,6.5,8.0]

--> combinations of n,T,W: 33*5*4
--> combinations of Lines/Tau: 6^4

valid_lines=['CO10','CO21','CO32','CO43',\
        'HCN10','HCN21','HCN32','HCN43',\
        'HCOP10','HCOP21','HCOP32','HCOP43',\
        'HNC10','HNC21','HNC32','HNC43']

-------------------------------------------------------------------------------------
------------------------------ CO models (9.2GB) ------------------------------------
-------------------------- min. RAM requirement: 32GB -------------------------------
-------------------------------------------------------------------------------------
shortname: co

The v2 CO models contain molecular line emissivities
for the following transitions and parameter space:

n, T, W...same as standard model
Tau=[0.1,0.2,0.3,
     5.0,6.5,8.0]

--> combinations of n,T,W: 33*5*4
--> combinations of Lines/Tau: 6^4

valid_lines=['CO10','CO21','CO32',\
        '13CO10','13CO21','13CO32',\
        'C18O10','C18O21','C18O32',\
        'C17O10','C17O21','C17O32']

-------------------------------------------------------------------------------------
---------------------------- Coarse models (22GB) -----------------------------------
------------------------- min. RAM requirement: 48GB --------------------------------
-------------------------------------------------------------------------------------
shortname:coarse

***********************
* NOT YET IMPLEMENTED *
***********************

The v2 coarse models contain molecular line emissivities
for the following transitions and parameter space:

n,W...same as standard model

T=[0,10,20,30]
Tau=[0.1,0.3,
     0.8,1.5,
     6.5,8.0]

--> combinations of n,T,W: 33*3*4
--> combinations of Lines/Tau: 6^7

valid_lines=['CO10','CO21','CO32',\
        'HCN10','HCN21','HCN32',\
        'HCOP10','HCOP21','HCOP32',\
        '13CO10','13CO21','13CO32',\
        'C18O10','C18O21','C18O32',\
        'HNC10','HNC21','HNC32',\
        'C17O10','C17O21','C17O32']

-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
