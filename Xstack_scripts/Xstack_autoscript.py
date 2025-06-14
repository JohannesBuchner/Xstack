#!/usr/bin/env python3
"""A comprehensive standalone pipeline code for X-ray spectral stacking.

X-ray spectral stacking is non-trivial compared to optical spectral stacking. The difficulties arise from two facts: 1) X-ray has much fewer photon counts (Poisson), meaning that spectral counts and uncertainties cannot be scaled simultaneously (as compared to optical); 2) X-ray has non-diagonal, complex response, meaning that the response needs to be taken into account when stacking.

To tackle these issues, we develop Xstack: a open-source, and comprehensive standalone pipeline code for X-ray spectral stacking. The methodology is to first sum all (rest-frame) PI spectra, without any scaling; and then sum the response files (ARFs and RMFs), each with appropriate weighting factors to preserve the overall spectral shape.

The key features of Xstack are: 1) properly account account for individual spectral contribution to the final stack, by assigning data-driven ARF weighting factors; 2) preserve Poisson statistics; 3) support Galactic absorption correction.


Examples
--------
Calling Xstack is simple. For this command line version, it is only a single-line task:
```
runXstack your_filelist.txt 1.0 2.3 SHP
```
And you will get `stack.pi`, `stackbkg.pi`, `stack.arf`, `stack.rmf`. Or more sophisticatedly:
```
runXstack your_filelist.txt 1.0 2.3 SHP --outsrc output_SRCname.pi --outbkg output_BKGname.pi --outrmf output_RMFname.rmf --outarf output_ARFname.arf --outfene output_FirstEnergyname.fene --nthreads 20 --ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf 
```
If you want to do bootstrap, that is also easy:
```
runXstack your_filelist.txt 1.0 2.3 SHP --outsrc output_SRCname.pi --outbkg output_BKGname.pi --outrmf output_RMFname.rmf --outarf output_ARFname.arf --outfene output_FirstEnergyname.fene --nthreads 20 --ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf --resample_method bootstrap --num_bootstrap 100 --resample_outdir YourOutputDir
```
Please see below for the documentation of each Args:

"""

# the main module
from Xstack import Xstack,rsXstack
# the usual packages
import numpy as np
from astropy.io import fits as pyfits
import os
import shutil
import sys
import argparse

nh_file = Xstack.default_nh_file
script_path = os.path.abspath(__file__) # the absolute path of Xstack_autoscript.py

class HelpfulParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write("error: %s\n" % message)
		self.print_help()
		sys.exit(2)

parser = HelpfulParser(description=__doc__,
	epilog="""Shi-Jiang Chen, Johannes Buchner and Teng Liu (C) 2025 <JohnnyCsj666@gmail.com>""",
    formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument("filelist", type=str, help="text file containing the file names")
parser.add_argument("--prefix", type=str, default="./results/stacked_", help="prefix for output stacked PI, BKGPI, ARF, and RMF files; defaults to './results/stacked_'")
# parser.add_argument("--outsrc", type=str, default="stack.pi", help="source PI output file name, or basename in resample mode")
# parser.add_argument("--outbkg", type=str, default="stackbkg.pi", help="Background PI output file name, or basename in resample mode")
# parser.add_argument("--outrmf", type=str, default="stack.rmf", help="RMF output file name, or basename in resample mode")
# parser.add_argument("--outarf", type=str, default="stack.arf", help="ARF output file name, or basename in resample mode")
# parser.add_argument("--outfene", type=str, default="fene.fits", help="name of output fits storing PI/ARF first energy, or basename in resample mode")
parser.add_argument("--flux_energy_lo", type=float, default=1.0, help="lower end of the energy range in keV for computing flux; defaults to 1.0")
parser.add_argument("--flux_energy_hi", type=float, default=2.3, help="upper end of the energy range in keV for computing flux; defaults to 2.3")
parser.add_argument("--rsp_weight_method", type=str, default="SHP", help="method to calculate RSP weighting factor for each source; 'SHP': assuming all sources have same spectral shape, 'FLX': assuming all sources have same shape and energy flux (weigh by exposure time), 'LMN': assuming all sources have same shape and luminosity (weigh by exposure/dist^2); defaults to 'SHP'")
parser.add_argument("--rsp_project_gamma", type=float, default=2.0, help="prior photon index value for projecting RSP matrix onto the output energy channel. This is used in the `SHP` method, to calculate the weight of each response; defaults to 2.0 (typical for AGN).")
parser.add_argument("--parametric_rmf", action="store_true", help="method to shift RMF: parametric (gaussian) or not (re-sample response matrices)")
parser.add_argument("--rm_ene_dsp", action="store_true", help="remove energy dispersion map (the map is used for shifting RMF in parametric mode)")
parser.add_argument("--nthreads", type=int, default=10, help="number of cpus used for non-parametric RMF shifting")
parser.add_argument("--num_bkg_groups", type=int, default=10, help="number of background groups")
parser.add_argument("--ene_trc", type=float, default=0.0, help="energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)")
parser.add_argument("--same_rmf", type=str, default=None, help="specify the name of common rmf, if all sources are to use the same rmf")
# adding bootstrap
parser.add_argument("--resample_method", type=str, default="None", help="method for performing resampling; 'None': no resampling, 'bootstrap': use bootstrap, 'KFold': use KFold)")
parser.add_argument("--num_bootstrap", type=int, default=10, help="number of bootstrap experiments")
parser.add_argument("--bootstrap_portion", type=float, default=1.0, help="portion of sources to resample in each bootstrap experiment")
parser.add_argument("--Ksort_filelist", type=str, default="Ksort_filelist.txt", help="name of file storing the sorting value for each source in `filelist`")
parser.add_argument("--K", type=int, default=4, help="number of groups for KFold")
# parser.add_argument("--resample_outdir", type=str, default="resample", help="name of output directory storing resampling files")


args = parser.parse_args()


def main():
    pifile_lst = []
    bkgpifile_lst = []
    arffile_lst = []
    rmffile_lst = []
    z_lst = []
    nh_lst = []

    for line in open(args.filelist):
        filename = line.rstrip()
        print("checking ...", filename)
        
        path = os.path.dirname(filename)
        a = pyfits.open(filename)
        header = a["SPECTRUM"].header
        #exposure = header["EXPOSURE"]
        #backscal = header["BACKSCAL"]
        #areascal = header["AREASCAL"]
        backfile = os.path.join(path, header["BACKFILE"])
        rmffile = os.path.join(path, header["RESPFILE"])
        # if all sources share the same RMF, you can manually point the `rmffile` to the common RMF"s path via --same_rmf argument:
        if args.same_rmf is not None:
              rmffile = args.same_rmf
        arffile = os.path.join(path, header["ANCRFILE"])

        b = pyfits.open(backfile)
        bheader = b["SPECTRUM"].header
        #bexposure = bheader["EXPOSURE"]
        #bbackscal = bheader["BACKSCAL"]
        #bareascal = bheader["AREASCAL"]
        #assert "RESPFILE" not in bheader or bheader["RESPFILE"] == header["RESPFILE"], "background must have same RMF"
        assert "ANCRFILE" not in bheader or bheader["ANCRFILE"] == header["ANCRFILE"], "background must have same ARF"
        
        z = float(open(filename + ".z").read())
        z_lst.append(z)
        nh = float(open(filename + ".nh").read())
        nh_lst.append(nh)
        pifile_lst.append(filename)
        bkgpifile_lst.append(backfile)
        arffile_lst.append(arffile)
        rmffile_lst.append(rmffile)


    if args.num_bkg_groups > len(pifile_lst):
          print("Warning! `Nbkggrp` must be smaller than the number of spectra loaded. `Nbkggrp` is now set to 1.")
          args.num_bkg_groups = 1


    if args.resample_method == "None":    # no resampling: single stack
        data = Xstack.XstackRunner(
            pifile_lst=pifile_lst,                          # the PI spectrum list
            arffile_lst=arffile_lst,                        # the ARF list
            rmffile_lst=rmffile_lst,                        # the RMF list
            z_lst=z_lst,                                    # the redshift list
            bkgpifile_lst=bkgpifile_lst,                    # the bkg PI files list
            nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                 # the source id list (optional)
            rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
            rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            rmfsft_method="PAR" if args.parametric_rmf else "NONPAR", # the RMF shifting method
            sample_rmf=None,                                # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                                # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                                # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                    # the number of background groups to calculate uncertainty of background
            ene_trc=args.ene_trc,                           # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
            rm_ene_dsp=args.rm_ene_dsp,                     # whether or not to remove the energy dispersion map
            nthreads=args.nthreads,                             # number of cpus used for RMF shifting
            prefix=args.prefix,                                 # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
            # o_pi_name=args.outsrc,                          # name of output PI spectrum file
            # o_bkgpi_name=args.outbkg,                       # name of output background PI spectrum file
            # o_arf_name=args.outarf,                         # name of output ARF file
            # o_rmf_name=args.outrmf,                         # name of output RMF file
            # o_fene_name=args.outfene,                       # name of output fenergy file
        ).run()
        print(data)


    elif args.resample_method == "bootstrap":     # resampling 
        data = rsXstack.resample_XstackRunner(
            pifile_lst=pifile_lst,                          # the PI spectrum list
            arffile_lst=arffile_lst,                        # the ARF list
            rmffile_lst=rmffile_lst,                        # the RMF list
            z_lst=z_lst,                                    # the redshift list
            bkgpifile_lst=bkgpifile_lst,                    # the bkg PI spectrum list
            nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                 # the source id list (optional)
            rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
            rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            rmfsft_method="PAR" if args.parametric_rmf else "NONPAR", # the RMF shifting method
            sample_rmf=None,                               # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                               # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                               # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                   # the number of background groups to calculate uncertainty of background
            ene_trc=args.ene_trc,                          # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
            rm_ene_dsp=args.rm_ene_dsp,                    # whether or not to remove the energy dispersion map
            nthreads=args.nthreads,                            # number of cpus used for RMF shifting
            resample_method=args.resample_method,          # resample method: `bootstrap` or `KFold`
            num_bootstrap=args.num_bootstrap,              # number of bootstrap experiments in `bootstrap` method
            bootstrap_portion=args.bootstrap_portion,      # portion to resample in `bootstrap` method
            prefix=args.prefix,                             # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
            # o_dir_name=args.resample_outdir,               # name of output directory to store all bootstrap files
            # o_pi_basename=args.outsrc,                     # basename of output PI spectrum files
            # o_bkgpi_basename=args.outbkg,                  # basename of output background PI spectrum files
            # o_arf_basename=args.outarf,                    # basename of output ARF files
            # o_rmf_basename=args.outrmf,                    # basename of output RMF files
            # o_fene_basename=args.outfene,                  # basename of output fenergy files
        ).run()
        print(data)


    elif args.resample_method == "KFold":     # resampling 
        Ksort_lst = []
        with open(args.Ksort_filelist) as f:
                lines = f.readlines()
        for line in lines:
                Ksort_lst.append(float(line.strip("\n")))
        Ksort_lst = np.array(Ksort_lst)
        assert len(Ksort_lst)==len(pifile_lst), "`Ksort_filelist` must have same length as `pifile_lst`!"

        data = rsXstack.resample_XstackRunner(
            pifile_lst=pifile_lst,                          # the PI spectrum list
            arffile_lst=arffile_lst,                        # the ARF list
            rmffile_lst=rmffile_lst,                        # the RMF list
            z_lst=z_lst,                                    # the redshift list
            bkgpifile_lst=bkgpifile_lst,                    # the bkg PI spectrum list
            nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                 # the source id list (optional)
            rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
            rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            rmfsft_method="PAR" if args.parametric_rmf else "NONPAR", # the RMF shifting method
            sample_rmf=None,                                # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                                # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                                # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                    # the number of background groups to calculate uncertainty of background
            ene_trc=args.ene_trc,                           # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
            rm_ene_dsp=args.rm_ene_dsp,                     # whether or not to remove the energy dispersion map
            nthreads=args.nthreads,                             # number of cpus used for RMF shifting
            resample_method=args.resample_method,           # resample method: `bootstrap` or `KFold`
            K=args.K,                                       # number of subgroups to divide the original sample into in `KFold` method
            Ksort_lst=Ksort_lst,                            # value list  used to sort the original sample in `KFold` method
            prefix=args.prefix,                             # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
            # o_dir_name=args.resample_outdir,                # name of output directory to store all bootstrap files
            # o_pi_basename=args.outsrc,                      # basename of output PI spectrum files
            # o_bkgpi_basename=args.outbkg,                   # basename of output background PI spectrum files
            # o_arf_basename=args.outarf,                     # basename of output ARF files
            # o_rmf_basename=args.outrmf,                     # basename of output RMF files
            # o_fene_basename=args.outfene,                   # basename of output fenergy files
        ).run()
        print(data)


    else:
        raise Exception("Available `resample_method`: `None`, `bootstrap`, or `KFold`!")


if __name__ == "__main__":
    main()

