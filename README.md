# Xstack
## What is <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> ?

<u><span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is a  comprehensive standalone pipeline code for **X-ray spectral shifting and stacking**.</u>

X-ray spectral stacking is **non-trivial** compared to optical spectral stacking. The difficulties arise from two facts:

1) X-ray has much fewer photon counts (Poisson), meaning that spectral **counts and uncertainties cannot be scaled simultaneously** (as compared to optical);
2) X-ray has non-diagonal, complex response, meaning that the **response needs to be taken into account** when stacking.

To tackle these issues, we develop <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>: a open-source, and comprehensive standalone pipeline code for X-ray spectral stacking. The methodology is to first sum all (rest-frame) PI spectra, without any scaling; and then sum the response files (ARFs and RMFs), each with appropriate weighting factors to preserve the overall spectral shape. The preservation of Poisson statistics for the data (and Gaussian for the background) ensures the validity of subsequent spectral fitting (via e.g., XSPEC).

## Key features of <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>

1) properly account for individual spectral contribution to the final stack, by assigning data-driven ARF weighting factors; 
2) preserve Poisson statistics; 
3) support Galactic absorption correction, if an additional ***NH*** value (in units of 1 $\text{cm}^{-2}$) for each spectrum is given.

## Prerequisites and Installation

To install <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> (along with the prerequisite packages), simply put:
```shell
git clone https://github.com/AstroChensj/Xstack.git
cd Xstack
python -m pip install .
```
`python -m` ensures that all required packages are installed for your current conda environment only (i.e., the path where your python is called, `which python`).

Troubleshooting:

- If you encounter network issues (e.g., `Port 443`) when installing with `https`, try `ssh` instead (see [this link](https://stackoverflow.com/questions/2643502/git-how-to-solve-permission-denied-publickey-error-when-using-git) for setup of github `ssh`):
  
  ```shell
  ssh -T git@github.com
  git clone git@github.com:AstroChensj/Xstack.git
  ```

## How to use <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>
Stacking X-ray spectra with <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is simple: you can either call it from command line, or invoke it as a module in python. In either case, <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> requires as input the individual source `PI` spectra, background `PI` spectra (with proper `BACKSCAL` parameters), effective area curve `ARF`s (extracted for source region), and response matrix `RMF`s. The output will be the stacked source `PI` spectrum, stacked background `PI` spectrum (already scaled), stacked `ARF`, and stacked `RMF`.

### 1. Command line version

- A simple and quick example:

  ```shell
  runXstack your_filelist.txt 1.0 2.3 SHP
  ```
  
  
  - And you will get the stacked spectra `stack.pi`, `stackbkg.pi`, and stacked response files `stack.arf`, `stack.rmf`. 
  
  - `runXstack` is the alias for `python3 /path/to/your/Xstack/scripts/Xstack_autoscript.py`, which should be set automatically after `python -m pip install .`.
  
  - `your_filelist.txt` stores the absolute path of the PI spectrum file for each source. The PI spectrum should follow OGIP standards – its header (of extension `SPECTRUM`) should have keywords helping <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> to find the corresponding bkg PI spectrum file (`BACKFILE`), the RMF (`RESPFILE`) and ARF (`ANCRFILE`).
  
  - `1.0` and `2.3`  are lower/upper end of the energy range (in keV) for computing flux. The flux represents the contribution from each source’s PI spectrum to the total stacked spectrum, and will be used as the weighting factors when stacking ARFs/RMFs.
  
  - `SHP` is the ARF weighting method, assuming all sources to be stacked have the same spectral shape (the minimum assumption). Under this method, the ARF weighting factor is calculated from flux (data-driven). 
  
- Or more sophisticatedly, specify more parameters:

  ```shell
  runXstack your_filelist.txt 1.0 2.3 SHP --outsrc output_SRCname.pi --outbkg output_BKGname.pi --outrmf output_RMFname.rmf --outarf output_ARFname.arf --outfene output_FirstEnergyname.fene --usecpu 20 --ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf 
  ```

  - `outsrc`  `outbkg`  `outrmf`  `outarf`  `outfene` specify the output stacked PI spectrum name, bkg PI spectrum name, RMF, ARF, and first energy file, respectively.

  -  `usecpu` specifies the number of CPUs used for shifting RMF.

  - `ene_trc` specifies the energy (keV) below which the ARF is unreliable and should manually be truncated. For example, for eROSITA there may be some calibration issues below 0.2 keV, so you can set this parameter to `0.2`.

  - `same_rmf` : the RMF files are usually large, and sometimes all sources to be stacked could share the same RMF in order to save space. Under this case, you can specify the file name of the common RMF with `same_rmf`.

- If you want to do bootstrap, that is also easy:

  ```shell
  runXstack your_filelist.txt 1.0 2.3 SHP --outsrc output_SRCname.pi --outbkg output_BKGname.pi --outrmf output_RMFname.rmf --outarf output_ARFname.arf --outfene output_FirstEnergyname.fene --usecpu 20 --ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf --resample_method bootstrap --num_bootstrap 100 --resample_outdir YourOutputDir
  ```

- You can run `runXstack -h` to get the documentation of all the above parameters. Or equivalently check below:

  | Parameters | Description | Default values|
  |---|---|---|
  |`filelist`|text file containing the file names|--|
  |`flux_energy_lo`|lower end of the energy range in keV for computing flux|--|
  |`flux_energy_hi`|upper end of the energy range in keV for computing flux|--|
  |`arf_scale_method`|method to calculate ARF weighting factor for each source; `FLX`: assuming all sources have same energy flux (weigh by exposure time), `LMN`: assuming all sources have same luminosity (weigh by exposure/dist^2), `SHP`: assuming all sources have same spectral shape|SHP|
  |`--outsrc`|source PI output file name, or basename in resample mode|stack.pi|
  |`--outbkg`|Background PI output file name, or basename in resample mode|stackbkg.pi|
  |`--outrmf`|RMF output file name, or basename in resample mode|stack.rmf|  
  |`--outarf`|ARF output file name, or basename in resample mode|stack.arf|
  |`--outfene`|name of output fits storing PI/ARF first energy, or basename in resample mod|stack.fene|
  |`--parametric_rmf`|use parametric method (time-saving yet crude short-cut) to shift RMF|--|
  |`--rm_ene_dsp`|remove energy dispersion map (the map is used for shifting RMF in parametric mode)|--|
  |`--usecpu`|number of cpus used for non-parametric RMF shifting|10|
  |`--num_bkg_groups`|number of background groups|10|
  |`--ene_trc`|energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)|0.0|
  |`--same_rmf`|specify the name of common rmf, if all sources are to use the same rmf|None|
  |`--resample_method`|method for performing resampling; `None`: no resampling, `bootstrap`: use bootstrap, `KFold`: use KFold)|None|
  |`--num_bootstrap`|number of bootstrap experiments in `bootstrap` mode|10|
  |`--bootstrap_portion`|portion of sources to resample in each bootstrap experiment|1.0|
  |`--Ksort_filelist`|name of file storing the sorting value for each source in `filelist`, under `KFold` mode|Ksort_filelist.txt|
  |`--K`|number of groups for `KFold`|4|
  |`--resample_outdir`|name of output directory storing resampling files|resample|

### 2.  Python module version
- An example:

  ```python
  from Xstack.Xstack import XstackRunner
  
  ## specify the input PIs, bkg PIs, RMFs, ARFs, redshifts, Galactic NHs ...
  #pifile_lst = [...]
  #bkgpifile_lst = [...]
  #rmffile_lst = [...]
  #arffile_lst = [...]
  #z_lst = [...]
  #nh_lst = [...]
  
  # run Xstack
  XstackRunner(
      pifile_lst=pifile_lst,                          # PI file list
      arffile_lst=arffile_lst,                        # ARF file list
      rmffile_lst=rmffile_lst,                        # RMF file list
      z_lst=z_lst,                                    # redshift list
      bkgpifile_lst=bkgpifile_lst,                    # bkg PI file list
      nh_lst=nh_lst,                                  # nh list
      arfscal_method='SHP',                           # method to calculate ARF weighting factor for each source (recommended: SHP)
      int_rng=(1.0,2.3),                              # if `arfscal_method`=`SHP`, choose the range to calculate flux
      rmfsft_method='NONPAR',                         # method to shift RMF
      nh_file=Xstack.default_nh_file,                 # the Galactic absorption profile (absorption factor vs. energy)
      Nbkggrp=10,                                     # the number of background groups to calculate uncertainty of background
      ene_trc=0.2,                                    # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
      usecpu=50,                                      # number of cpus used for RMF shifting
      o_pi_name='stack.pi',              				# name of output PI spectrum file
      o_bkgpi_name='stackbkg.pi',       				# name of output background PI spectrum file
      o_arf_name='stack.arf',            				# name of output ARF file
      o_rmf_name='stack.rmf',            				# name of output RMF file
      fene_name='stack.fene',            				# name of output fenergy file
  ).run()
  ```

  - see `help(XstackRunner)` for the documentation for each input parameter.

- Or bootstrap:

  ```py
  from Xstack.Xstack import resample_XstackRunner
  
  ## specify the input PIs, bkg PIs, RMFs, ARFs, redshifts, Galactic NHs ...
  #pifile_lst = [...]
  #bkgpifile_lst = [...]
  #rmffile_lst = [...]
  #arffile_lst = [...]
  #z_lst = [...]
  #nh_lst = [...]
  
  # run Xstack
  resample_XstackRunner(
      pifile_lst=pifile_lst,                          # PI file list
      arffile_lst=arffile_lst,                        # ARF file list
      rmffile_lst=rmffile_lst,                        # RMF file list
      z_lst=z_lst,                                    # redshift list
      bkgpifile_lst=bkgpifile_lst,                    # bkg PI file list
      nh_lst=nh_lst,                                  # nh list
      arfscal_method='SHP',                           # method to calculate ARF weighting factor for each source (recommended: SHP)
      int_rng=(1.0,2.3),                              # if `arfscal_method`=`SHP`, choose the range to calculate flux
      rmfsft_method='NONPAR',                         # method to shift RMF
      nh_file=Xstack.default_nh_file,                 # the Galactic absorption profile (absorption factor vs. energy)
      Nbkggrp=10,                                     # the number of background groups to calculate uncertainty of background
      ene_trc=0.2,                                    # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
      usecpu=50,                                      # number of cpus used for RMF shifting
      resample_method='bootstrap',              		# resample method: `bootstrap` or `KFold`
      num_bootstrap=20,							    # number of bootstrap experiments in `bootstrap` method
      bootstrap_portion=1.0,							# portion to resample in `bootstrap` method
      o_dir_name='resample',							# name of output directory to store all bootstrap files
      o_pi_name='stack.pi',              				# basename of output PI spectrum file
      o_bkgpi_name='stackbkg.pi',       				# basename of output background PI spectrum file
      o_arf_name='stack.arf',            				# basename of output ARF file
      o_rmf_name='stack.rmf',            				# basename of output RMF file
      fene_name='stack.fene',            				# basename of output fenergy file
  ).run()
  ```

  - see `help(resample_XstackRunner)` for the documentation for each input parameter.

- View [`./demo/demo.ipynb`](https://nbviewer.org/github/AstroChensj/Xstack/blob/main/demo/demo.ipynb) for a quick walk-through and more examples!
