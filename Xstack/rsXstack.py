##############################################
############# MAIN FUNCTION ##################
##############################################
import numpy as np
from astropy.io import fits
import shutil
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib._parallel_backends import LokyBackend
from .Xstack import XstackRunner

######## DEPRECATED USAGE #########
# class NestedBackend(LokyBackend):
#     def get_nested_backend(self):
#         backend = NestedBackend()
#         backend.nested_level = 0
#         return backend, None
###################################

class resample_XstackRunner:
    """
    A batch of `XstackRunner` objects. Used for producing bootstrap (or K-Fold) stacked spectra.

    Example usage
    -------------
    data = resample_XstackRunner(
        pifile_lst = your_pifile_lst,
        arffile_lst = your_arffile_lst,
        rmffile_lst = your_rmffile_lst,
        z_lst = your_z_lst,
        bkgpifile_lst = your_bkgpifile_lst,
        o_dir_name = "bootstrap",
        o_pi_basename = "src.fits",
        o_bkgpi_basename = "bkg.fits",
        o_arf_basename = "arf.fits",
        o_rmf_basename = "rmf.fits",
        o_fene_basename = "fene.fits",
        # and other arguments if you like
    )
    data.run()  # this will produce a batch of bootstrap stacked PIs, bkgPIs, ARFs, RMFs under `bootstrap` directory 
    """
    def __init__(
            self,pifile_lst,arffile_lst,rmffile_lst,z_lst,bkgpifile_lst=None,nh_lst=None,srcid_lst=None,rspwt_method="SHP",rspproj_gamma=2.0,int_rng=(1.0,2.3),rmfsft_method="NONPAR",sample_rmf=None,sample_arf=None,nh_file=None,Nbkggrp=10,ene_trc=None,rm_ene_dsp=False,usecpu=1,resample_method="bootstrap",num_bootstrap=10,bootstrap_portion=1.0,K=4,Ksort_lst=None,o_dir_name=None,o_pi_basename=None,o_bkgpi_basename=None,o_arf_basename=None,o_rmf_basename=None,o_fene_basename=None
        ):
        """
        Parameters
        ----------
        pifile_lst : list or numpy.ndarray
            The input PI spectrum file list.
        arffile_lst : list or numpy.ndarray
            The input ARF file list.
        rmffile_lst : list or numpy.ndarray
            The input RMF file list.
        z_lst : list or numpy.ndarray
            The redshift list.
        bkgpifile_lst : list or numpy.ndarray, optional
            The input background PI spectrum list. Defaults to None.
        nh_lst : list or numpy.ndarray, optional
            The Galactic absorption column density list (in units of 1 cm^{-2}). Defaults to None.
        srcid_lst : list or numpy.ndarray, optional
            The source ID list. Defaults to None.
        rspwt_method : str, optional
            Method for calculating ARFSCAL. Defaults to `SHP`. Available methods are:
            - `SHP`: assuming all sources have same spectral shape, recommended
            - `FLX`: assuming all sources have same flux (erg/s/cm^2)
            - `LMN`: assuming all sources have same luminosity (erg/s)
        rspproj_gamma : float, optional
            The prior photon index value for projecting RSP matrix onto the output energy channel. This is used in the `SHP` method, to calculate the weight of each response. Defaults to 2.0 (typical for AGN).
        int_rng : tuple of (float,float), optional
            The energy (keV) range for computing flux. Defaults to (1.0,2.3).
        rmfsft_method : str, optional
            The RMF shifting method. Defaults to `NONPAR`. Two methods are available:
            - `NONPAR`: Non-PARameterized method, i.e. shift the probability profile directly. This should be more accurate, and takes into account the off-diagonal elements in the RMF matrix. However, the non-PARameterized method is more time-consuming than PARameterized method (~10^2 times slower).
            - `PAR`: Parameterized method, i.e. approximate the probability profile with a Gaussian, and shift the Gaussians.
        sample_rmf : str, optional
            Name of sample RMF. Defaults to None.
        sample_arf : str, optional
            Name of sample ARF. Defaults to None.
        nh_file : str, optional
            Galactic absorption profile (absorption factor vs. energy) at 1e20 cm^{-2}. If specified, galactic absorption correction will be applied on the ARF before shifting.
            - Should be in txt format. 
            - Should also contain the following columns in the first extension: `nhene_ce`, `nhene_wd`, `factor`.
            - `factor` should indicate the absorption factor when nh=1e20.
            - An easy way to obtain the `nh_file`: iplot `tbabs*powerlaw` with `Nh`=1e20 and `PhoIndex`=0.0, `Norm`=1 in Xspec.
        Nbkggrp : int, optional
            Number of groups with similar background-to-source scaling ratio. Defaults to 10.
        ene_trc : float, optional
            Truncate energy below which manually set ARF and PI counts to zero. For eROSITA, `ene_trc` is typically set as 0.2 keV. Defaults to None.
        rm_ene_dsp : bool, optional
            Whether or not to remove the energy dispersion map at each run. Generating dispersion map could take some time. Defaults to False.
        usecpu : int, optional
            Number of CPUs used in shifting RSP.
        resample_method : str, optional
            Resampling method. Defaults to `bootstrap`. Available methods are:
            - `bootstrap`: Resample a certain portion (default 1.0; modified by `bootstrap_portion`) of original sample with replacement, for `num_bootstrap` times.
            - `KFold`: First sort the sample according to some values (specified by `Ksort_lst`), then leave out 1/K fraction from start to end for K iterations.
        num_bootstrap : int, optional
            Number of bootstrap experiments. Defaults to 10.
        bootstrap_portion : float, optional
            Portion of original sample in each bootstrap experiment. Defaults to 1.0.
        K : int, optional
            Number of subgroups to divide the original sample into in `KFold` method. Defaults to 4.
        Ksort_lst : list or numpy.ndarray, optional
            The value list (same length as the original sample) used to sort the original sample in `KFold` method.
        o_dir_name : str, optional
            Name of output directory to store all bootstrap files.
        o_pi_basename : str, optional
            Basename of output PI spectrum files. Defaults to None (do not produce output files). E.g., if `o_pi_basename`=stack.pi, then you will get resampled PI files like stack01.pi, stack02.pi, ... .
        o_bkgpi_basename : str, optional
            Basename of output background PI spectrum files. Defaults to None (do not produce output files).
        o_arf_basename : str, optional
            Basename of output ARF files. Defaults to None (do not produce output files).
        o_rmf_basename : str, optional
            Basename of output RMF files. Defaults to None (do not produce output files).
        o_fene_basename : str, optional
            Basename of output fenergy files. Defaults to None (do not produce output files).
        """
        self.pifile_lst = np.array(pifile_lst)
        self.arffile_lst = np.array(arffile_lst)
        self.rmffile_lst = np.array(rmffile_lst)
        self.z_lst = np.array(z_lst)
        self.bkgpifile_lst = np.array(bkgpifile_lst)
        assert len(pifile_lst)==len(arffile_lst)==len(rmffile_lst)==len(z_lst)==len(bkgpifile_lst), f"The input `pifile_lst`({len(pifile_lst)}), `arffile_lst`({len(arffile_lst)}), `rmffile_lst`({len(rmffile_lst)}), `z_lst`({len(z_lst)}), and `bkgpifile_lst`({len(bkgpifile_lst)}) must have same shape! "
        if nh_lst is not None:
            self.nh_lst = np.array(nh_lst)
        else:
            self.nh_lst = np.zeros(len(pifile_lst))
        if srcid_lst is not None:
            self.srcid_lst = np.array(srcid_lst)
        else:
            self.srcid_lst = np.arange(len(pifile_lst))
        self.rspwt_method = rspwt_method
        self.rspproj_gamma = rspproj_gamma
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
        if Nbkggrp > len(pifile_lst):
            print("Warning! `Nbkggrp` must be smaller than the number of spectra loaded. `Nbkggrp` is now set to 1.")
            self.Nbkggrp = 1
        else:
            self.Nbkggrp = Nbkggrp
        self.ene_trc = ene_trc
        self.rm_ene_dsp = rm_ene_dsp
        self.usecpu = usecpu
        self.resample_method = resample_method
        if resample_method not in ["bootstrap","KFold"]:
            raise Exception("`resample_method` invalid (allowed: `bootstrap`, `KFold`) !")
        if resample_method == "bootstrap":
            self.num_bootstrap = num_bootstrap
            self.bootstrap_portion = bootstrap_portion
        if resample_method == "KFold":
            if Ksort_lst is None:
                raise Exception("Since you have chosen `KFold` as `resample_method`, you need to specify `Ksort_lst` !")
            assert len(Ksort_lst) == len(pifile_lst), "The `Ksort_lst` must have same shape as `pifile_lst`! "
            self.Ksort_lst = np.array(Ksort_lst)
            self.K = K
        
        if o_dir_name is None:
            self.o_dir_name = resample_method
        else:
            self.o_dir_name = o_dir_name
        self.o_pi_basename = o_pi_basename
        self.o_bkgpi_basename = o_bkgpi_basename
        self.o_arf_basename = o_arf_basename
        self.o_rmf_basename = o_rmf_basename
        self.o_fene_basename = o_fene_basename

        self.XstackRunner_lst = []

    def run(self):
        """
        Shift all PIs + bkgPIs + ARFs + RMFs to rest-frame in one go.
        """
        # shutil.rmtree("%s"%self.o_dir_name,ignore_errors=True)
        # os.mkdir("%s"%self.o_dir_name) # make a directory to store all bootstrapped stacked pi, bkgpi, ARF and RMF files
        os.system(f"mkdir -p {self.o_dir_name}")
        if self.resample_method == "bootstrap":
            np.random.seed(self.num_bootstrap) # initialize seed
            for i in range(self.num_bootstrap):
                idx = str(i).zfill(len(str(self.num_bootstrap)))
                sampled_idx = np.random.choice(np.arange(len(self.pifile_lst)),size=int(len(self.pifile_lst)*self.bootstrap_portion),replace=True)
                sampled_pifile_lst = self.pifile_lst[sampled_idx]
                sampled_bkgpifile_lst = self.bkgpifile_lst[sampled_idx]
                sampled_arffile_lst = self.arffile_lst[sampled_idx]
                sampled_rmffile_lst = self.rmffile_lst[sampled_idx]
                sampled_z_lst = self.z_lst[sampled_idx]
                sampled_nh_lst = self.nh_lst[sampled_idx]
                sampled_srcid_lst = self.srcid_lst[sampled_idx]
                o_pi_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_pi_basename)[0]}_{idx}{os.path.splitext(self.o_pi_basename)[1]}"
                o_bkgpi_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_bkgpi_basename)[0]}_{idx}{os.path.splitext(self.o_bkgpi_basename)[1]}"
                o_arf_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_arf_basename)[0]}_{idx}{os.path.splitext(self.o_arf_basename)[1]}"
                o_rmf_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_rmf_basename)[0]}_{idx}{os.path.splitext(self.o_rmf_basename)[1]}"
                o_fene_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_fene_basename)[0]}_{idx}{os.path.splitext(self.o_fene_basename)[1]}"

                XstackRunner_i = XstackRunner(
                    pifile_lst=sampled_pifile_lst,
                    arffile_lst=sampled_arffile_lst,
                    rmffile_lst=sampled_rmffile_lst,
                    z_lst=sampled_z_lst,
                    bkgpifile_lst=sampled_bkgpifile_lst,
                    nh_lst=sampled_nh_lst,
                    srcid_lst=sampled_srcid_lst,
                    rspwt_method=self.rspwt_method,
                    rspproj_gamma=self.rspproj_gamma,
                    int_rng=self.int_rng,
                    rmfsft_method=self.rmfsft_method,
                    sample_rmf=self.sample_rmf,
                    sample_arf=self.sample_arf,
                    nh_file=self.nh_file,
                    Nbkggrp=self.Nbkggrp,
                    ene_trc=self.ene_trc,
                    rm_ene_dsp=self.rm_ene_dsp,
                    usecpu=self.usecpu,
                    o_pi_name=o_pi_name_i,
                    o_bkgpi_name=o_bkgpi_name_i,
                    o_arf_name=o_arf_name_i,
                    o_rmf_name=o_rmf_name_i,
                    o_fene_name=o_fene_name_i,
                )
                self.XstackRunner_lst.append(XstackRunner_i)

            # Parallel(n_jobs=self.num_bootstrap,backend=NestedBackend())(delayed(self.process_entry)(i) for i in tqdm(range(self.num_bootstrap)))
            for i in range(self.num_bootstrap):
                self.XstackRunner_lst[i].run()
                print(f"****************** Current iteration: #{i+1}/{self.num_bootstrap} ********************")

        elif self.resample_method == "KFold":
            sorted_idx = np.argsort(self.Ksort_lst)
            sortedpifile_lst = self.pifile_lst[sorted_idx]
            sortedarffile_lst = self.arffile_lst[sorted_idx]
            sortedrmffile_lst = self.rmffile_lst[sorted_idx]
            sortedz_lst = self.z_lst[sorted_idx]
            sortedbkgpifile_lst = self.bkgpifile_lst[sorted_idx]
            sortednh_lst = self.nh_lst[sorted_idx]
            sortedsrcid_lst = self.srcid_lst[sorted_idx]
            Nsource = len(sortedpifile_lst)

            for i in range(self.K):
                idx = str(i).zfill(len(str(self.K)))
                mask = (np.arange(Nsource)<int(i*Nsource/self.K)) | (np.arange(Nsource)>=int((i+1)*Nsource/self.K))
                sampled_sortedpifile_lst = sortedpifile_lst[mask]
                sampled_sortedarffile_lst = sortedarffile_lst[mask]
                sampled_sortedrmffile_lst = sortedrmffile_lst[mask]
                sampled_sortedz_lst = sortedz_lst[mask]
                sampled_sortedbkgpifile_lst = sortedbkgpifile_lst[mask]
                sampled_sortednh_lst = sortednh_lst[mask]
                sampled_sortedsrcid_lst = sortedsrcid_lst[mask]
                o_pi_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_pi_basename)[0]}_{idx}{os.path.splitext(self.o_pi_basename)[1]}"
                o_bkgpi_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_bkgpi_basename)[0]}_{idx}{os.path.splitext(self.o_bkgpi_basename)[1]}"
                o_arf_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_arf_basename)[0]}_{idx}{os.path.splitext(self.o_arf_basename)[1]}"
                o_rmf_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_rmf_basename)[0]}_{idx}{os.path.splitext(self.o_rmf_basename)[1]}"
                o_fene_name_i = f"{self.o_dir_name}/{os.path.splitext(self.o_fene_basename)[0]}_{idx}{os.path.splitext(self.o_fene_basename)[1]}"

                XstackRunner_i = XstackRunner(
                    pifile_lst=sampled_sortedpifile_lst,
                    arffile_lst=sampled_sortedarffile_lst,
                    rmffile_lst=sampled_sortedrmffile_lst,
                    z_lst=sampled_sortedz_lst,
                    bkgpifile_lst=sampled_sortedbkgpifile_lst,
                    nh_lst=sampled_sortednh_lst,
                    srcid_lst=sampled_sortedsrcid_lst,
                    rspwt_method=self.rspwt_method,
                    rspproj_gamma=self.rspproj_gamma,
                    int_rng=self.int_rng,
                    rmfsft_method=self.rmfsft_method,
                    sample_rmf=self.sample_rmf,
                    sample_arf=self.sample_arf,
                    nh_file=self.nh_file,
                    Nbkggrp=self.Nbkggrp,
                    ene_trc=self.ene_trc,
                    rm_ene_dsp=self.rm_ene_dsp,
                    usecpu=self.usecpu,
                    o_pi_name=o_pi_name_i,
                    o_bkgpi_name=o_bkgpi_name_i,
                    o_arf_name=o_arf_name_i,
                    o_rmf_name=o_rmf_name_i,
                    o_fene_name=o_fene_name_i,
                )
                self.XstackRunner_lst.append(XstackRunner_i)

            for i in range(self.K):
                self.XstackRunner_lst[i].run()
                print(f"****************** Current iteration: #{i+1}/{self.K} ********************")

        else:
            raise Exception("`resample_method` invalid (allowed: `bootstrap`, `KFold`) !")
        
        return
    
    ######## DEPRECATED USAGE #########
    # def process_entry(self,i):
    #     self.Xstack_lst[i].run()
    #     return
    ###################################