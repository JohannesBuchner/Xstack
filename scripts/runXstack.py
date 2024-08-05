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
#import numpy as np
from astropy.io import fits as pyfits
import os
import sys
import argparse

nh_file = Xstack.default_nh_file

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
parser.add_argument('--num_bkg_groups', type=int, default=10, help='number of background groups')

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

    # initialize Xstack
    data_po = Xstack.XSTACK(
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
        rm_ene_dsp=True,                               # whether or not to remove the energy dispersion map
        o_pha_name=args.outsrc,
        o_bkgpha_name=args.outbkg,
        o_arf_name=args.outarf,
        o_rmf_name=args.outrmf,
    ).run()
    print(data_po)

if __name__ == '__main__':
    main()

