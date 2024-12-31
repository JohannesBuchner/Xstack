#!/usr/bin/env python3
"""


A brief overview of Xstack:

Xstack is a open-source, light-weight code for X-ray spectral shifting and stacking.

Given the redshift of each spectrum in a list, Xstack will first shift these spectra (PHA, counts vs. output energy (channel)) from observed-frame to rest-frame, and then sum them together. The response matrix files (RMF, the probability that a photon with input energy
will be detected with an output energy ) and ancillary response files (ARF, effective area vs. input energy) are shifted and stacked in a similar way to the spectrum. 

Xstack also supports correction of Galactic absorption, if an additional NH value (in units of 1) for each spectrum is given.

"""

# the main module
from Xstack import Xstack

# some miscellaneous packages for plotting, and generating fake spectra
#from Xstack.misc import rebin_pha, rebin_arf, make_grpflg, make_fkspec
#from Xstack.shift_arf import align_arf

# the usual packages
import numpy as np
from astropy.io import fits as pyfits
import os
import shutil
import sys
import argparse

nh_file = Xstack.default_nh_file
script_path = os.path.abspath(__file__) # the absolute path of runXstack.py

class HelpfulParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

parser = HelpfulParser(description=__doc__,
	epilog="""Shi-Jiang Chen, Johannes Buchner and Teng Liu (C) 2024 <JohnnyCsj666@gmail.com>""",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('filelist', type=str, help='text file containing the file names')
parser.add_argument('--outsrc', type=str, default='stack.pi', help='Source PHA output file name')
parser.add_argument('--outbkg', type=str, default='stackbkg.pi', help='Background PHA output file name')
parser.add_argument('--outrmf', type=str, default='stack.rmf', help='RMF output file name')
parser.add_argument('--outarf', type=str, default='stack.arf', help='ARF output file name')
parser.add_argument('flux_energy_lo', type=float, help='lower end of the energy range in keV for computing flux')
parser.add_argument('flux_energy_hi', type=float, help='upper end of the energy range in keV for computing flux')
parser.add_argument('arf_scale_method', type=str, default='SHP', help='method to calculate ARF weighting factor for each source: "FLX": assuming all sources have same energy flux (weigh by exposure time), "LMN": assuming all sources have same luminosity (weigh by exposure/dist^2), "SHP": assuming all sources have same spectral shape')
parser.add_argument('--parametric_rmf', type=bool, default=True, help='method to shift RMF: parametric (gaussian) or not (re-sample response matrices)')
parser.add_argument('--rm_ene_dsp', action='store_true', help='remove energy dispersion map (the map is used for shifting RMF in parametric mode)')
parser.add_argument('--num_bkg_groups', type=int, default=10, help='number of background groups')
# adding bootstrap
parser.add_argument('--num_bootstrap', type=int, default=-1, help='Whether or not to perform bootstrap experiments. If `num_bootstrap`>0, will generate `num_bootstrap` scripts, which can be later performed by the user with parallel command such as `parallel -j 10 -a bootstrap.sh`. `num_bootstrap` is by default -1, which means that no bootstrap experiments will be performed.')
parser.add_argument('--pre_bootstrap', type=str, default='bs', help='prefix of all bootstrap files')


args = parser.parse_args()


def main():
    phafile_lst = []
    bkgphafile_lst = []
    arffile_lst = []
    rmffile_lst = []
    z_lst = []
    nh_lst = []

    for line in open(args.filelist):
        filename = line.rstrip()
        print("checking ...", filename)
        
        path = os.path.dirname(filename)
        a = pyfits.open(filename)
        header = a['SPECTRUM'].header
        #exposure = header['EXPOSURE']
        #backscal = header['BACKSCAL']
        #areascal = header['AREASCAL']
        backfile = os.path.join(path, header['BACKFILE'])
        rmffile = os.path.join(path, header['RESPFILE'])
        arffile = os.path.join(path, header['ANCRFILE'])

        b = pyfits.open(backfile)
        bheader = b['SPECTRUM'].header
        #bexposure = bheader['EXPOSURE']
        #bbackscal = bheader['BACKSCAL']
        #bareascal = bheader['AREASCAL']
        assert 'RESPFILE' not in bheader or bheader['RESPFILE'] == header['RESPFILE'], 'background must have same RMF'
        assert 'ANCRFILE' not in bheader or bheader['ANCRFILE'] == header['ANCRFILE'], 'background must have same ARF'
        
        z = float(open(filename + '.z').read())
        z_lst.append(z)
        nh = float(open(filename + '.nh').read())
        nh_lst.append(nh)
        phafile_lst.append(filename)
        bkgphafile_lst.append(backfile)
        arffile_lst.append(arffile)
        rmffile_lst.append(rmffile)


    if args.num_bkg_groups > len(phafile_lst):
          print('Warning! `Nbkggrp` must be smaller than the number of spectra loaded. `Nbkggrp` is now set to 1.')
          args.num_bkg_groups = 1

    # Experimental feature: bootstrap (activated by setting num_bootstrap>0)
    if args.num_bootstrap > 0:
          shutil.rmtree('%s_filelist'%args.pre_bootstrap,ignore_errors=True)
          shutil.rmtree('%s_data'%args.pre_bootstrap,ignore_errors=True)
          os.mkdir('%s_filelist'%args.pre_bootstrap) # make a directory to store the bootstrapped filelists
          os.mkdir('%s_data'%args.pre_bootstrap) # make a directory to store all bootstrapped stacked PHA, bkgPHA, ARF and RMF files
          command_lst = []
          np.random.seed(args.num_bootstrap) # initialize seed
          for i in range(args.num_bootstrap):
               idx = str(i).zfill(len(str(args.num_bootstrap)))
               sampled_idx = np.random.choice(np.arange(len(phafile_lst)),size=len(phafile_lst),replace=True)
               sampled_phafile_lst = np.array(phafile_lst)[sampled_idx]
               filelist_txt = '%s_filelist/%s_%s_filelist.txt'%(args.pre_bootstrap,args.pre_bootstrap,idx)
               with open(filelist_txt,'w') as f:
                     for j in range(len(sampled_idx)):
                           f.writelines('%s\n'%sampled_phafile_lst[j])
               command_lst.append('python '+script_path
                                  +' '+filelist_txt
                                  +' '+str(args.flux_energy_lo)
                                  +' '+str(args.flux_energy_hi)
                                  +' '+args.arf_scale_method
                                  +' --outsrc=%s_data/%s_%s_%s'%(args.pre_bootstrap,args.pre_bootstrap,idx,args.outsrc)
                                  +' --outbkg=%s_data/%s_%s_%s'%(args.pre_bootstrap,args.pre_bootstrap,idx,args.outbkg)
                                  +' --outrmf=%s_data/%s_%s_%s'%(args.pre_bootstrap,args.pre_bootstrap,idx,args.outrmf)
                                  +' --outarf=%s_data/%s_%s_%s'%(args.pre_bootstrap,args.pre_bootstrap,idx,args.outarf)
                                  +' --parametric_rmf='+str(args.parametric_rmf)
                                  # keep ene_dsp_map, so --rm_ene_dsp is not included
                                  +' --num_bkg_groups='+str(args.num_bkg_groups)
                                  +' --num_bootstrap=-1'  # deactivate boostrap in each individual run
                                  )
          bs_script = '%s.sh'%args.pre_bootstrap
          with open(bs_script,'w') as f:
                for i in range(len(command_lst)):
                      f.writelines('%s\n'%command_lst[i])
          # then the user can run `bs_script` by parallel command
          # e.g. parallel -j 20 -a bs_script
          # 20: number of cores to be used
    
    # Normal case (bootstrap deactivated by setting num_bootstrap<=0)
    else:   
        # initialize Xstack and run
        data = Xstack.XSTACK(
            phafile_lst=phafile_lst,                       # the PHA files list
            arffile_lst=arffile_lst,                       # the ARF list
            rmffile_lst=rmffile_lst,                       # the RMF list
            z_lst=z_lst,                                   # the redshift list
            bkgphafile_lst=bkgphafile_lst,                 # the bkg PHA files list
            nh_lst=nh_lst,                                 # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                # the source id list (optional)
            arfscal_method=args.arf_scale_method,          # the method to calculate ARF weighting factor for each source
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            rmfsft_method='PAR' if args.parametric_rmf else 'NONPAR', # the RMF shifting method
            sample_rmf=None,                               # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                               # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                               # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                   # the number of background groups to calculate uncertainty of background
            rm_ene_dsp=args.rm_ene_dsp,                    # whether or not to remove the energy dispersion map
            o_pha_name=args.outsrc,
            o_bkgpha_name=args.outbkg,
            o_arf_name=args.outarf,
            o_rmf_name=args.outrmf,
        ).run()
        print(data)

if __name__ == '__main__':
    main()

