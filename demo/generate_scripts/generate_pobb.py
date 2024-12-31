import numpy as np
from Xstack.simulate_spec import make_fkspec
import os
from joblib import Parallel,delayed
from tqdm import tqdm

model = 'pobb'
Nspec = 200
np.random.seed(1)   # set seed for np.random
usecpu = 20

base_dir = os.getcwd() + '/data/' + model
out_dir = os.getcwd() + '/data/' + model + '/' + model + '_spec'
log_dir = os.getcwd() + '/data/' + model + '/' + model + '_log'
os.system('rm -rf %s'%out_dir)
os.system('rm -rf %s'%log_dir)

rand = np.random.rand(Nspec*4)
z_lst = 0.3 + 1.7*rand[:Nspec]
tlum_int_lst = 44.5 + rand[Nspec:2*Nspec]
gamma_lst = 1.9 + 0.2*rand[2*Nspec:3*Nspec]
q_lst = 0.4*rand[3*Nspec:]  # soft excess strength
srcid_lst = np.arange(Nspec)
srcid_lst = np.array(['{:05d}'.format(srcid) for srcid in srcid_lst])
out_src_file_lst = []


def process_entry(i):
    z = z_lst[i]
    gamma = gamma_lst[i]
    tlum_int = tlum_int_lst[i]
    q = q_lst[i]
    srcid = srcid_lst[i]
    expo = 2000
    seed = i
    out_pre = srcid + '_' + model

    spec_dir = os.getcwd() + '/data'
    src_file = os.getcwd() + '/data/sample.src'
    arf_file = os.getcwd() + '/data/sample.arf'
    rmf_file = os.getcwd() + '/data/sample.rmf'

    make_fkspec(model,[z,tlum_int,gamma,q],seed,spec_dir,src_file,rmf_file,arf_file,out_pre,out_dir,log_dir,expo,expo)

    out_src_file = out_dir + '/' + out_pre + '.pi'

    return out_src_file


out_src_file_lst = Parallel(n_jobs=usecpu)(delayed(process_entry)(i) for i in tqdm(range(len(tlum_int_lst))))

with open(base_dir+'/'+model+'_filelist.txt','w') as f:
    for out_src_file in out_src_file_lst:
        f.writelines(out_src_file+'\n')
