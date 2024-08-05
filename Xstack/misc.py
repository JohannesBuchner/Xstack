import numpy as np
from astropy.io import fits
import os
import re
from scipy.constants import c
import astropy.units as u
from astropy.cosmology import Planck18
import sfdmap
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from .shift_arf import align_arf


#===================================================
##################### grppha #######################
#===================================================
def make_grpflg(src_name,grp_name,method='EDGE',rmf_file='',eelo=None,eehi=None):
    '''
    Add `GROUPING` column to the source PHA file.
    
    Available Methods
    -----------------
    * `EDGE`: Group by fixed energy bin edges.
    
    Parameters
    ----------
    src_name: Input source PHA file name.
    grp_name: Output grouped PHA file name.
    method: Grouping method.
    rmf_file: (for `EDGE` method) RMF file name. If not specified, the code will 
              automatically search the header of `src_name`.
    eelo: (for `EDGE` method) Lower edge of fixed energy bin.
    eehi: (for `EDGE` method) Upper edge of fixed energy bin.
    
    Returns
    -------
    grpflg: `GROUPING` column written in `grp_name`.
    '''
    if method == 'EDGE':
        if (eelo is None) or (eehi is None):
            raise Exception('Please specify `eelo` and `eehi` in method `EDGE`!')
        # find channel energy in EBOUNDS extension of RMF file
        with fits.open(src_name) as hdu:
            data = hdu['EBOUNDS'].data
            head = hdu['EBOUNDS'].header
            chan = data['CHANNEL']
            try:
                src_rmf = head['RESPFILE']
            except Exception as e:
                src_rmf = ''
        
        # the RMF file must either be specified as `rmf_file`, or specified in the header of `src_name`
        if os.path.exists(rmf_file):
            pass
        elif os.path.exists(src_rmf):
            rmf_file = src_rmf
        else:
            raise Exception('Either the RMF file is not specified as `rmf_file`, or the one in %s does not exist!'%src_name)
        
        with fits.open(rmf_file) as hdu:
            ebo = hdu['EBOUNDS'].data
        ene_lo = ebo['E_MIN']
        ene_hi = ebo['E_MAX']
        ene_ce = (ene_lo + ene_hi) / 2
        
        assert len(chan)==len(ene_ce), 'CHANNEL and RMF EBOUNDS ENERGY does not match!'
        assert np.all(eelo<eehi)==True, '`eelo` has to be smaller than `eehi`!'
        
        # make grouping flag
        grpflg = np.ones(len(ene_ce))
        eece = (eelo + eehi) / 2
        eeid = np.arange(len(eece))
        eeid_bk = [-1] # stores the energy id that has been used
        for i in range(len(ene_ce)):
            mask = (ene_ce[i]<=eehi) & (ene_ce[i]>eelo)
            if np.all(mask==False): # outside eelo~eehi
                grpflg[i] = 1
                continue
            if eeid[mask] > max(eeid_bk): # step to a new bin
                grpflg[i] = 1
            else:
                grpflg[i] = -1
            eeid_bk.append(eeid[mask])
        
        # create output file
        # for Linux system:
        os.system('cp -f %s %s'%(src_name,grp_name))
        # for Windows system, uncomment the line below (also remember to change `/` to `\`)
        #os.system('copy %s %s'%(src_name,grp_name))
        with fits.open(grp_name,mode='update') as hdu:
            SPECTRUM = hdu['SPECTRUM']
            if 'GROUPING' in SPECTRUM.columns.names:
                SPECTRUM.columns.del_col('GROUPING')    # remove 'GROUPING' column if it exists beforehand
            GROUPING = fits.Column(name='GROUPING', format='I', array=grpflg)
            SPECTRUM.data = fits.BinTableHDU.from_columns(SPECTRUM.columns + GROUPING).data
        
        return grpflg
    
    else:
        raise Exception('Available method for grppha: EDGE!')


def rebin_pha(ene_lo,ene_hi,coun,coun_err,grpflg):
    '''
    Rebin PHA file according to `grpflg`.
    
    Parameters
    ----------
    ene_lo : array_like
        Lower edge of energy bin.
    ene_hi : array_like
        Upper edge of energy bin.
    coun : array_like
        Photon counts in each channel.
    coun_err : array_like
        Photon counts error in each channel.
    grpflg : array_like
        Grouping flag. Must have same length as `ene_lo` or `ene_hi`.
    
    Returns
    -------
    grpene_lo : ndarray
        Lower edge of grouped energy bin.
    grpene_hi : ndarray
        Upper edge of grouped energy bin.
    grpcoun : ndarray
        Photon counts in each grouped energy bin.
    grpcoun_err : ndarray
        Photon counts error in each grouped energy bin.
    '''
    ene_ce = (ene_lo + ene_hi) / 2
    assert len(grpflg) == len(ene_ce), 'grpflag shape '+str(grpflg.shape)+' does not match ene shape '+str(ene_ce.shape)+' !'
    
    grpene_lo = []
    grpene_hi = []
    grpcoun = []
    grpcoun_err = []
    
    tmpene_lo = []
    tmpene_hi = []
    tmpcoun = []
    tmpcoun_err = []
    
    for i in range(len(ene_ce)):
        if grpflg[i] == 1:    # start of group
            # collect data
            # if ene_tmp_lst is empty (usually the case for the first energy bin), just skip this step
            if len(tmpene_lo)!=0:
                grpene_lo.append(tmpene_lo[0])
                grpene_hi.append(tmpene_hi[-1])
                grpcoun.append(np.sum(tmpcoun))
                grpcoun_err.append(np.sqrt(np.sum(np.array(tmpcoun_err)**2)))
            tmpene_lo = [ene_lo[i]]
            tmpene_hi = [ene_hi[i]]
            tmpcoun = [coun[i]]
            tmpcoun_err = [coun_err[i]]
        elif grpflg[i] == -1:    # continuing of group
            tmpene_lo.append(ene_lo[i])
            tmpene_hi.append(ene_hi[i])
            tmpcoun.append(coun[i])
            tmpcoun_err.append(coun_err[i])
        else: 
            raise Exception('`grpflg` not in standard format (`1` for start of group, `-1` for continuing of group)')
    
    # for the last energy bin
    grpene_lo.append(tmpene_lo[0])
    grpene_hi.append(tmpene_hi[-1])
    grpcoun.append(np.sum(tmpcoun))
    grpcoun_err.append(np.sqrt(np.sum(np.array(tmpcoun_err)**2)))
    
    grpene_lo = np.array(grpene_lo)
    grpene_hi = np.array(grpene_hi)
    grpcoun = np.array(grpcoun)
    grpcoun_err = np.array(grpcoun_err)
        
    return grpene_lo,grpene_hi,grpcoun,grpcoun_err


def rebin_arf(arfene_lo,arfene_hi,specresp,ene_lo,ene_hi,coun,grpflg):
    '''
    
    '''
    ene_ce = (ene_lo + ene_hi) / 2
    specresp_ali = align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp)

    grpene_lo = []
    grpene_hi = []
    grpspecresp = []

    tmpene_lo = []
    tmpene_hi = []
    tmpspecresp = []
    tmpwt = []    # weight

    for i in range(len(ene_ce)):
        if grpflg[i] == 1:    # start of group
            # collect data
            # if ene_tmp_lst is empty (usually the case for the first energy bin), just skip this step
            if len(tmpene_lo)!=0:
                grpene_lo.append(tmpene_lo[0])
                grpene_hi.append(tmpene_hi[-1])
                tmpspecresp = np.array(tmpspecresp)
                tmpwt = np.array(tmpwt)
                grpspecresp.append((tmpspecresp * tmpwt / tmpwt.sum()).sum())
            tmpene_lo = [ene_lo[i]]
            tmpene_hi = [ene_hi[i]]
            tmpspecresp = [specresp_ali[i]]
            tmpwt = [coun[i]/specresp_ali[i] if coun[i]>0 else 0]   # caution! may be refined later
        elif grpflg[i] == -1:    # continuing of group
            tmpene_lo.append(ene_lo[i])
            tmpene_hi.append(ene_hi[i])
            tmpspecresp.append(specresp_ali[i])
            tmpwt.append(coun[i]/specresp_ali[i] if coun[i]>0 else 0)   # caution! may be refined later
        else: 
            raise Exception('`grpflg` not in standard format (`1` for start of group, `-1` for continuing of group)')
        
    # for the last energy bin
    grpene_lo.append(tmpene_lo[0])
    grpene_hi.append(tmpene_hi[-1])
    tmpspecresp = np.array(tmpspecresp)
    tmpwt = np.array(tmpwt)
    grpspecresp.append((tmpspecresp * tmpwt / tmpwt.sum()).sum())
    
    grpene_lo = np.array(grpene_lo)
    grpene_hi = np.array(grpene_hi)
    grpspecresp = np.array(grpspecresp)
        
    return grpene_lo,grpene_hi,grpspecresp


#===================================================
################# UV correction ####################
#===================================================
def dered(flux,RA,DEC,R=5.28):
    '''
    
    Parameters
    ----------
    flux : float
        Flux to be dereddened.
    RA : float
        Right Ascension of the source.
    DEC : float
        Declination of the source.
    R : float
        Extinction coefficient of the band in use (see e.g. Fang+2023)

    Returns
    -------
    flux_der : float
        The dereddened flux.
    '''
    # Assuming R(UVW1)=5.28 (Page+2013)
    m = sfdmap.SFDMap() # Assuming that SFD_DIR has been written in ~/.bashrc as an environmental variable!
    A = R * m.ebv(RA,DEC)
    flux_der = flux * 10**(A/2.5)
    return flux_der

def kcorr(flux,z,alpha=0.65):
    '''
    
    Parameters
    ----------
    flux : float
        The observed-frame flux (erg/cm^2/s/AA) recorded by some filter.
    z : float
        Redshift of the source.
    alpha : float
        Spectral slope (assuming F_nu ~ nu^{-alpha}, where F_nu in units of erg/cm^2/s/AA).
    
    Returns
    -------
    flux_int : float
        The expected flux (erg/cm^2/s/AA) recorded by some filter, if observed in rest-frame. 
        Note that the distance effect (F_nu ~ L_nu / 4/pi/d_L^2) has to be considered separately.
    '''
    # UV SED: F_nu~nu^-alpha
    # Assume alpha=0.65 (Natali+1998)
    K = 2.5*(alpha-1)*np.log10(1+z)
    flux_int = flux * 10**(0.4*K)
    return flux_int

def flux_sft(flux,lambda_fr=2315.7,lambda_to=2500,alpha=0.65):
    '''
    Calculate the expected flux (erg/cm^2/s/Hz) at `lambda_to`, if we already know the flux at 
    `lambda_fr`.
    
    Parameters
    ----------
    flux : float
    lambda_fr : float
        lambda (from)
    lambda_to : float
        lambda (to)
    alpha : float
        Spectral slope (assuming F_nu ~ nu^{-alpha}, where F_nu in units of erg/cm^2/s/Hz).
    '''
    # UV SED: F_nu~nu^-alpha
    # Assume alpha=0.65 (Natali+1998)
    flux_sft = flux * (lambda_to / lambda_fr)**alpha
    return flux_sft

def flux2lum(flux,z):
    '''
    
    Parameters
    ----------
    flux : float
        Monochromatic flux (erg/cm^2/s/Hz).
    z : float
        Redshift.

    Returns
    -------
    lum : float
        Monochromatic luminosity (erg/s/Hz).
    '''
    luminosity_distance = Planck18.luminosity_distance(z)
    # Calculate the luminosity using the formula: Luminosity = 4 * pi * (D_L^2) * F_observed / (1 + z)
    # Where Luminosity is in erg/s, D_L is the luminosity distance, F_observed is the observed flux, and z is the redshift.
    #luminosity = 4 * np.pi * (luminosity_distance.to(u.cm).value ** 2) * flux / (1 + z)
    lum = 4 * np.pi * (luminosity_distance.to(u.cm).value ** 2) * flux
    return lum

def restlum(flux,z,alpha,lambda_fr=2315.7,lambda_to=2500,R=None,RA=0,DEC=0):
    '''
    
    Parameters
    ----------
    flux : float
        Observed-frame monochromatic flux (erg/cm^2/s/Hz).
    z : float
    alpha : float
    lambda_fr : float
        Lambda (from) in units of AA.
    lambda_to : float
        Lambda (to) in units of AA.
    
    Returns
    -------
    lum : float
        Rest-frame luminosity, nu*F_nu or lambda*F_lambda (erg/s)
    '''
    # deredden (optional)
    if R is not None:
        flux = dered(flux,RA,DEC,R) # erg/cm^2/s/Hz
    # K correction
    flux = kcorr(flux,z,alpha) # erg/cm^2/s/Hz
    # Shift to some common band (e.g. 2500AA)
    flux = flux_sft(flux,lambda_fr,lambda_to,alpha) # erg/cm^2/s/Hz
    flux = flux * c / (lambda_to * 1e-10) # erg/cm^2/s
    # Flux to Lum
    lum = flux2lum(flux,z) # erg/s
    return lum


#===================================================
################# GalNH (X-ray) ####################
#===================================================
def get_nh(RA,DEC):
    '''
    
    Parameters
    ----------
    RA : float
    DEC : float

    Returns
    -------
    nh_val : float
        nh values in units of 1 cm^-2
    '''
    # write sh
    log_file = 'nh.log'
    os.system('rm -rf %s'%log_file)
    shell_file = open('run_nh.sh','w',newline='')
    shell_file.writelines('(\n')
    shell_file.writelines('echo 2000\n') # Equinox (d/f 2000)
    shell_file.writelines('echo %f\n'%RA) # RA in hh mm ss.s or degrees
    shell_file.writelines('echo %f\n'%DEC) # DEC in hh mm ss.s or degrees
    shell_file.writelines(') | nh\n')
    shell_file.close()
    # run sh
    os.system("bash run_nh.sh > %s 2>&1"%log_file)
    # read sh
    with open(log_file,'r') as file:
        text = file.read()
    # Use regular expression to find the line with 'Weighted' and capture the value
    match1 = re.search(r'Weighted average nH \(cm\*\*-2\)\s+([0-9.E+-]+)', text)
    match2 = re.search(r'h1_nh_HI4PI.fits >> nH \(cm\*\*-2\)\s+([0-9.E+-]+)', text) # in case when the given RA/DEC falls outside the allowed range
    if match1:
        nh_val = match1.group(1)
    elif match2:
        nh_val = match2.group(1)
    else:
        raise Exception('Invalid RA (%.4f), DEC(%.4f) for nh!'%(RA,DEC))
    nh_val = float(nh_val)

    return nh_val


#===================================================
############### RMF Visualization ##################
#===================================================
def view_rmf(rmf_file,fig_name,n_grid_i=1000,n_grid=1000):
    '''
    
    Parameters
    ----------
    rmf_file : str
        Name of the RMF file.
    fig_name : str
        Output figure name.
    n_grid_i : int
        Number of grids for the input energy (does not have to be the same as the length of `ENERG_LO` or `ENERG_HI`), default 1000.
    n_grid : int
        Number of grids for the output channel energy (does not have to be the same as the length of `E_MIN` or `E_MAX`), default 1000.

    Returns
    -------
    None.
    '''
    with fits.open(rmf_file) as hdu:
        rmf = hdu['MATRIX'].data # to determine the input (=arf) energy bin (ENERG_LO,ENERG_HI)
        ebo = hdu['EBOUNDS'].data # to determine the output (channel) energy bin (E_MIN,E_MAX)
    
    iene_lo = rmf['ENERG_LO'] # input energy lower edge
    iene_hi = rmf['ENERG_HI'] # input energy upper edge
    iene_ce = (iene_lo + iene_hi) / 2
    
    ene_lo = ebo['E_MIN'] # output energy lower edge
    ene_hi = ebo['E_MAX'] # output energy upper edge
    ene_ce = (ene_lo + ene_hi) / 2
    
    grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
    prob = np.zeros(grid[0].shape)
    
    n_grp = rmf['N_GRP']
    f_chan = rmf['F_CHAN']
    n_chan = rmf['N_CHAN']
    mat = rmf['MATRIX']
    
    f_chan_0 = int(f_chan.min()) # the zero point of channel index
    for i in range(len(iene_ce)):
        f_mat = 0
        for grp_j in range(n_grp[i]):
            f_chan_j = f_chan[i][grp_j] - f_chan_0
            n_chan_j = n_chan[i][grp_j]
            e_chan_j = f_chan_j + n_chan_j # ending index of group_j in channel
            e_mat = f_mat + n_chan_j # ending index of group_j in matrix[i]
            
            prob[i][f_chan_j:e_chan_j] += mat[i][f_mat:e_mat]
            f_mat += n_chan_j
            
    # The energy bin width may not be uniform
    # e.g. smaller energy bin width near 0.05 keV, but larger energy bin width near 16 keV
    # For better visualization, we do 2d-interpolation!
    interp = RegularGridInterpolator((iene_ce, ene_ce), prob,
                                     bounds_error=False, fill_value=None)
    
    iene_new = np.linspace(min(iene_lo),max(iene_hi),n_grid_i+1)
    iene_lo_new = iene_new[:-1]
    iene_hi_new = iene_new[1:]
    iene_ce_new = (iene_lo_new + iene_hi_new) / 2
    
    ene_new = np.linspace(min(ene_lo),max(ene_hi),n_grid+1)
    ene_lo_new = ene_new[:-1]
    ene_hi_new = ene_new[1:]
    ene_ce_new = (ene_lo_new + ene_hi_new) / 2
    
    grid_new = np.meshgrid(ene_ce_new,iene_ce_new)
    prob_new = interp((grid_new[1],grid_new[0]))
    
    # normalize each row
    row_sum = np.sum(prob_new,axis=1)
    prob_new = prob_new / row_sum[:,np.newaxis]
    
    fig, (ax1) = plt.subplots(1,1,figsize=(4,4))
    
    ax1.imshow(prob_new[::-1],
              extent=(ene_ce_new[0],ene_ce_new[-1],iene_ce_new[0],iene_ce_new[-1]),
              aspect='auto', cmap='gray_r',)
    
    ax1.set_xlabel('Output Channel Energy (keV)',fontsize=15)
    ax1.tick_params("x",which="major",
                length=10,width = 1.0,size=5,labelsize=10,pad=3)
    ax1.tick_params("x",which="minor",
                length=10,width = 1.0,size=3,labelsize=10,pad=3)
    
    ax1.set_ylabel('Input Energy (keV)',fontsize=15)
    ax1.tick_params("y",which="major",
                  length=10,width = 1.0,size=5,labelsize=10)
    ax1.tick_params("y",which="minor",
                  length=10,width = 1.0,size=3,labelsize=10)
    
    spines = ax1.spines
    for spine in spines.values():
        spine.set_linewidth(2.5)
        
    plt.savefig('%s'%fig_name,bbox_inches='tight',transparent=False,dpi=300)

    return


#===================================================
##################### XSPEC ########################
#===================================================
def pygrppha(src_name,grp_name,grpmin=25):
    with open('grppha.sh','w') as shell_file:
        shell_file.writelines('rm -rf %s\n'%(grp_name))
        shell_file.writelines('(\n')
        shell_file.writelines('echo %s\n'%(src_name))
        shell_file.writelines('echo %s\n'%(grp_name))
        shell_file.writelines('echo group min %d\n'%(grpmin))
        shell_file.writelines('echo exit\n')
        shell_file.writelines(') | grppha\n')
    os.system("bash grppha.sh > grppha.log 2>&1")
    return


#===================================================
################## SIMULTION #######################
#===================================================
def make_fkspec(spec_model,par_lst,seed,spec_dir,pha_file,arf_file,out_pre,out_dir,log_dir,src_expo=None,bkg_expo=None,run_sh=True,pipeline_file=None):
    '''
    Make fake spectrum.

    Parameters
    ----------
    spec_model : str
        The spectral model.
        * `po` : single powerlaw
        * `abspo` : single absorbed powerlaw
        * `pobb` : primary continuum (powerlaw) + soft excess (bbody)
        * `abspobb` : absorbed primary continuum (powerlaw) + soft excess (bbody)
        * `popo` : primary continuum (powerlaw) + soft excess (powerlaw)
        * `abspopo` : absorbed primary continuum (powerlaw) + soft excess (powerlaw)
    
    par_lst : array_like
        The parameter list.
        * `po` : z, PC lum (0.5-10), gamma
        * `abspo` : nh, z, PC lum (0.5-10), gamma
        * `pobb` : z, PC lum (0.5-2), gamma, SE strength q (0.5-2)
        * `abspobb` : nh, z, PC lum (0.5-2), gamma, se strength q (0.5-2)
        * `popo` : z, PC lum (0.5-2), gamma, SE strength q (0.5-2), SE gamma
        * `abspopo` : nh, z, PC lum (0.5-2), gamma, SE strength q (0.5-2), SE gamma
    
    seed : int
        Seed for `fakeit` in XSPEC.

    spec_dir : str
        The directory of your sample input SRC spectrum, BKG spectrum, ARF and RMF.

    pha_file : str
        The name of the SRC spectrum.

    arf_file : str
        The name of your sample input ARF file.

    out_pre : str
        The prefix of all output files (SRC spectrum, BKG spectrum, ARF, RMF).

    log_dir : str
        The directory to store the log file.

    src_expo : float or None
        Source exposure time for the fake spectrum (optional). If None, exposure time from `pha_file` will be used. Default is None.

    bkg_expo : float or None
        Background exposure time for the fake spectrum (optional). If None, exposure time from `pha_file` will be used. Default is None.

    run_sh : bool or None
        Whether to generate fake spectrum immediately. Default is True.
        * True : generate fake spectrum immediately
        * False: append a shell command to a shell script file (`pipeline_file`), which can be run later with parallel
    
    pipeline_file : str or None
        The shell script file. Default is None
    
    Returns
    -------
    None
    
    '''
    if src_expo is None:
        src_expo = ' '
    else:
        src_expo = str(src_expo)
    if bkg_expo is None:
        bkg_expo = ' '
    else:
        bkg_expo = str(bkg_expo)
    os.system('mkdir -p %s'%out_dir)
    os.system('mkdir -p %s'%log_dir)
    if spec_model == 'po':
        sh_file = 'spec_sh/po.sh'
    elif spec_model == 'abspo':
        sh_file = 'spec_sh/abspo.sh'
    elif spec_model == 'pobb':
        sh_file = 'spec_sh/pobb.sh'
    elif spec_model == 'abspobb':
        sh_file = 'spec_sh/abspobb.sh'
    elif spec_model == 'popo':
        sh_file = 'spec_sh/popo.sh'
    elif spec_model == 'abspopo':
        sh_file = 'spec_sh/abspopo.sh'
    else:
        raise Exception('Available spec_model: `po`, `abspo`, `pobb`, `abspobb`, `popo`, `abspopo` !')
    par_lst = [str(ele) for ele in par_lst]
    par_str = ' '.join(par_lst)
    if run_sh == True:  # if you want to generate the spectrum right away
        os.system("bash %s %s %d %s %s %s %s %s %s %s > %s/%s.log 2>&1"%(sh_file,par_str,seed,spec_dir,pha_file,arf_file,out_pre,out_dir,src_expo,bkg_expo,log_dir,out_pre))
    else:   # if you only want to generate the shell script (for later use, e.g. run with parallel)
        os.system("echo 'sh %s %s %d %s %s %s %s %s %s %s > %s/%s.log 2>&1' >> %s"%(sh_file,par_str,seed,spec_dir,pha_file,arf_file,out_pre,out_dir,src_expo,bkg_expo,log_dir,out_pre,pipeline_file))
    return

def pofluxconv(flux1,start1,end1,start2,end2,alpha):
    '''
    
    Parameters
    ----------
    flux1 : float
        original flux (erg/s/cm^2)

    start1 : float
        starting energy for original flux

    end1 : float
        ending energy for original flux

    start2 : float
        starting energy for new flux

    end2 : float
        ending energy for new flux

    alpha : float
        spectral slope (F_E~E^{-alpha})

    Returns
    -------
    flux2 : float
        the converted flux (erg/s/cm^2)

    '''
    flux2 = flux1 * (start2**(-alpha+1) - end2**(-alpha+1))/(start1**(-alpha+1) - end1**(-alpha+1))
    return flux2
