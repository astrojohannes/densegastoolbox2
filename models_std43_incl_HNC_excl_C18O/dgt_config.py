#!/usr/bin/env python

# available in neff_distrib files:
# grid_T=[10,15,20,25,30]
# grid_width=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# grid_tau=[0.05,0.1,0.2,0.3,0.8,0.9,1.0,1.1,1.2,1.5,2.0,3.0,4.0,5.0,5.5,5.8,5.9,6.0,6.1,6.2,6.5,7.0,7.5,8.0,9.0,10.]


# Down-selected for performance reasons
valid_T=[0,10,15,20,25,30]      # 5
valid_W=[0,0.2,0.4,0.6,0.8]     # 4
valid_Tau=[0.1,0.2,0.3,0.8,1.1,1.5,5.0,6.5,8.0] # 9

# 5 species
valid_lines=['CO10','CO21','CO32','CO43',\
        'HCN10','HCN21','HCN32','HCN43',\
        'HCOP10','HCOP21','HCOP32','HCOP43',\
        '13CO10','13CO21','13CO32','13CO43',\
#        'C18O10','C18O21','C18O32','C18O43',\
        'HNC10','HNC21','HNC32','HNC43']
#        'C17O10','C17O21','C17O32','C17O43',]
#        'CS10','CS21','CS32','CS43']
