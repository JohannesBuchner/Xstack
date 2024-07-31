##############################################
############# MAIN FUNCTION ##################
##############################################
from Xstack.shift_pha import *
from Xstack.shift_arf import *
from Xstack.shift_rmf import *

import numpy as np
from astropy.io import fits
from tqdm import tqdm

class XSTACK:
    '''
    X-ray Spectral Shifting & Stacking.
    '''
    def __init__(self,phafile_lst,arffile_lst,rmffile_lst,z_lst,
                 bkgphafile_lst=None,nh_lst=None,srcid_lst=None,arfscal_method='SHP',int_rng=(1.0,2.3),rmfsft_method='PAR',sample_rmf=None,sample_arf=None,nh_file=None,Nbkggrp=10,rm_ene_dsp=False,
                 o_pha_name=None,o_bkgpha_name=None,o_arf_name=None,o_rmf_name=None,fenergy_name=None):
        '''
        Parameters
        ----------
        phafile_lst : 
        arffile_lst : 
        rmffile_lst : 
        z_lst : 
        bkgphafile_lst : 
        nh_lst : 
        srcid_lst : 
        arfscal_method : 
        int_rng : 
        rmfsft_method : 
        sample_rmf : 
        sample_arf : 
        nh_file : 
        Nbkggrp : 
        o_pha_name : 
        o_bkgpha_name : 
        o_arf_name : 
        o_rmf_name : 
        fenergy_name : 
        '''
        self.phafile_lst = phafile_lst
        self.arffile_lst = arffile_lst
        self.rmffile_lst = rmffile_lst
        self.z_lst = z_lst
        self.bkgphafile_lst = bkgphafile_lst
        if nh_lst is not None:
            self.nh_lst = nh_lst
        else:
            self.nh_lst = np.zeros(len(phafile_lst))
        if srcid_lst is not None:
            self.srcid_lst = srcid_lst
        else:
            self.srcid_lst = np.arange(len(phafile_lst))
        self.arfscal_method = arfscal_method
        self.int_rng = int_rng
        self.rmfsft_method = rmfsft_method
        if sample_rmf is None:
            self.sample_rmf = rmffile_lst[0]
        else:
            self.sample_rmf = sample_rmf
        if sample_arf is None:
            self.sample_arf = arffile_lst[0]
        else:
            self.sample_arf = sample_arf
        self.nh_file = nh_file
        self.Nbkggrp = Nbkggrp
        self.o_pha_name = o_pha_name
        self.o_bkgpha_name = o_bkgpha_name
        self.o_arf_name = o_arf_name
        self.o_rmf_name = o_rmf_name
        self.fenergy_name = fenergy_name

        # creating empty lists to store results
        self.pha_sft_lst = []
        self.arf_sft_lst = []
        self.rmf_sft_lst = []
        self.bkgpha_sft_lst = []

        self.bkgscal_lst = []
        self.expo_lst = []
        self.first_energy_lst = []

    def run(self):
        '''
        Shift all PHAs + bkgPHAs + ARFs + RMFs to rest-frame in one go.
        '''
        with fits.open(self.sample_rmf) as hdu:
            mat = hdu['MATRIX'].data
            ebo = hdu['EBOUNDS'].data
        ene_lo = ebo['E_MIN']
        ene_hi = ebo['E_MAX']
        iene_lo = mat['ENERG_LO']
        iene_hi = mat['ENERG_HI']
        
        # SHIFTING
        print('################### Shifting ... ######################')
        for i in tqdm(range(len(self.srcid_lst))):
            phafile = self.phafile_lst[i]
            bkgphafile = self.bkgphafile_lst[i]
            arffile = self.arffile_lst[i]
            rmffile = self.rmffile_lst[i]
            z = self.z_lst[i]
            nh = self.nh_lst[i]
            srcid = self.srcid_lst[i]

            # PHA shifting
            (pha_chan_sft,pha_coun_sft,pha_chan,pha_coun) = shift_pha(phafile,self.sample_rmf,z)
            self.pha_sft_lst.append(pha_coun_sft.astype('float64'))
            self.first_energy_lst.append(ene_lo[pha_chan_sft[pha_coun_sft>0][0]])
            # BKGPHA shifting
            if self.bkgphafile_lst is None:
                bkgpha_coun_sft = np.zeros(len(pha_coun_sft))
            else:
                (bkgpha_chan_sft,bkgpha_coun_sft,bkgpha_chan,bkgpha_coun) = shift_pha(bkgphafile,self.sample_rmf,z)
            self.bkgpha_sft_lst.append(bkgpha_coun_sft.astype('float64'))
            # ARF shifting
            arf_sft = shift_arf(arffile,z,nh_file=self.nh_file,nh=nh)
            self.arf_sft_lst.append(arf_sft)
            # RMF shifting
            if self.rmfsft_method=='NONPAR':
                with fits.open(rmffile) as hdu:
                    mat = hdu['MATRIX'].data
                    ebo = hdu['EBOUNDS'].data
            rmf_sft = shift_rmf(mat,ebo,z,rmfsft_method=self.rmfsft_method)
            self.rmf_sft_lst.append(rmf_sft)

            # EXPO & BKGSCAL
            self.bkgscal_lst.append(get_bkgscal(phafile,bkgphafile))
            self.expo_lst.append(get_expo(phafile))

        # STACKING
        print('################### Stacking ... ######################')
        expo = np.sum(self.expo_lst)
        pha_stk,phaerr_stk = add_pha(self.pha_sft_lst,fits_name=self.o_pha_name,expo=expo,bkg_file=self.o_bkgpha_name,rmf_file=self.o_rmf_name,arf_file=self.o_arf_name)
        bkgpha_stk,bkgphaerr_stk = add_bkgpha(self.bkgpha_sft_lst,bkgscal_lst=self.bkgscal_lst,Ngrp=self.Nbkggrp,fits_name=self.o_bkgpha_name,expo=expo)
        arf_stk = add_arf(self.arf_sft_lst,self.pha_sft_lst,self.z_lst,bkgpha_lst=self.bkgpha_sft_lst,bkgscal_lst=self.bkgscal_lst,
                           ene_lo=ene_lo,ene_hi=ene_hi,arfene_lo=iene_lo,arfene_hi=iene_hi,
                           expo_lst=self.expo_lst,int_rng=self.int_rng,arfscal_method=self.arfscal_method,fits_name=self.o_arf_name,sample_arf=self.sample_arf,srcid_lst=self.srcid_lst)
        rmf_stk = add_rmf(self.rmf_sft_lst,self.o_arf_name,expo_lst=self.expo_lst,fits_name=self.o_rmf_name,sample_rmf=self.sample_rmf,srcid_lst=self.srcid_lst)
        
        first_energy_fits(self.srcid_lst,self.first_energy_lst,self.fenergy_name)
        
        return pha_stk, phaerr_stk, bkgpha_stk, bkgphaerr_stk, arf_stk, rmf_stk


    