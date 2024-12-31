import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18
import astropy.units as u
from tqdm import tqdm
import os


def shift_arf(arf_file,z,nh_file=None,nh=1e20,ene_trc=None):
    '''
    Shift ARF in the same way as the PI file. Galactic absorption correction will be applied on the ARF before shifting,
    if `nh_file` is specified.

    In the shifted ARF:
    The lower energy edges and higher energy edges remain the same as the unshifted ARF, but specresp is taken as the 
    value that corresponds to the nearest shifted energy bin.
    e.g. specresp_sft@2keV = specresp@1keV, if redshift=1

    Parameters
    ----------
    arf_file : str
        The name of ARF.
    z : float
        Redshift of the source.
    nh_file : str, optional
        Galactic absorption profile (absorption factor vs. energy). If specified, galactic absorption 
        correction will be applied on the ARF before shifting.
        - Should be in txt format. 
        - Should also contain the following columns in the first extension: `nhene_ce`, `nhene_wd`, `factor`.
        - `factor` should indicate the absorption factor when nh=1e20.
        - An easy way to obtain the `nh_file`: iplot `tbabs*powerlaw` with `Nh`=1e20 and `PhoIndex`=0.0, `Norm`=1 in Xspec.
    nh : float, optional
        The galactic absorption nh of the source (e.g. 3e20). Defaults to 1e20.
    ene_trc : float, optional
        Truncate energy below which manually set ARF and PI counts to zero. For eROSITA, `ene_trc` is typically 0.2 keV. Defaults to None.

    Returns
    -------
    specresp_sft : numpy.ndarray
        The shifted (and also galactic absorption corrected) ARF specresp (cm^2 vs. arf energy).
    '''

    with fits.open(arf_file) as hdu:
        arf = hdu['SPECRESP'].data    # SPECRESP extension
    arfene_lo = arf['ENERG_LO']
    arfene_hi = arf['ENERG_HI']
    arfene_ce = (arfene_lo + arfene_hi) / 2
    arfene_wd = arfene_hi - arfene_lo
    specresp = arf['SPECRESP']

    # do galactic absorption correction (optional)
    if nh_file is not None:
        with open(nh_file,'r') as file:
            lines = file.readlines()
        nhene_ce = []
        nhene_wd = []
        factor = []
        for line in lines:
            nhene_ce.append(float(line.split(' ')[0]))
            nhene_wd.append(float(line.split(' ')[1]))
            factor.append(float(line.split(' ')[2]))
        nhene_ce = np.array(nhene_ce)
        nhene_wd = np.array(nhene_wd)
        nhene_lo = nhene_ce - nhene_wd
        nhene_hi = nhene_ce + nhene_wd
        factor = np.array(factor)
        specresp = corr_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh)
    
    arfene_sft_lo = arfene_lo * (1+z)
    arfene_sft_hi = arfene_hi * (1+z)
    arfene_sft_ce = arfene_ce * (1+z)
    arfene_sft_wd = arfene_wd * (1+z)

    # truncate below ene_trc
    if ene_trc is not None:
        idx_trc = np.argmin(abs(arfene_ce-ene_trc))
        specresp[:idx_trc] = 0
    
    # In the shifted ARF:
    # The lower energy edges and higher energy edges remain the same as the unshifted ARF
    # But specresp is taken as the value that corresponds to the nearest shifted energy bin
    # e.g. specresp_sft@2keV = specresp@1keV, if redshift=1
    specresp_sft = np.zeros(len(specresp))
    for i in range(len(specresp_sft)):
        mask = (arfene_lo[i] <= arfene_sft_hi) & (arfene_hi[i] >= arfene_sft_lo)
        if np.all(mask==False):
            continue
        arfene_mask_lo = arfene_sft_lo[mask].copy()
        arfene_mask_hi = arfene_sft_hi[mask].copy()
        arfene_mask_ce = arfene_sft_ce[mask].copy()
        arfene_mask_wd = arfene_sft_wd[mask].copy()
        specresp_mask = specresp[mask].copy()
        
        # for the first and last channel in the basket, we need to recalculate their widths
        arfene_mask_wd[0] = arfene_mask_hi[0] - arfene_lo[i]
        arfene_mask_wd[-1] = arfene_hi[i] - arfene_mask_lo[-1]
        
        prob_mask = arfene_mask_wd / arfene_mask_wd.sum()
        specresp_sft[i] = (specresp_mask * prob_mask).sum()
        
    return specresp_sft


def add_arf(specresp_lst,pi_lst,z_lst,bkgpi_lst=None,bkgscal_lst=None,ene_lo=None,ene_hi=None,arfene_lo=None,arfene_hi=None,
            expo_lst=None,int_rng=(1.0,2.3),arfscal_method='SHP',fits_name=None,sample_arf='sample.arf',srcid_lst=None,prob_lst=None):
    '''
    Weighted sum of ARF specresp.

    Parameters
    ----------
    specresp_lst : list or numpy.ndarray
        The ARF specresp (cm^2 vs. arf energy) list.
    pi_lst : list or numpy.ndarray
        The PI list.
    z_lst : list or numpy.ndarray
        The redshift list.
    bkgpi_lst : list or numpy.ndarray, optional
        The background PI list. Defaults to None.
    bkgscal_lst : list or numpy.ndarray, optional
        The background scaling-ratio list. Defaults to None.
    ene_lo : numpy.ndarray, optional
        Lower edge of output channel energy bin. Defaults to None.
    ene_hi : numpy.ndarray, optional
        Upper edge of output channel energy bin. Defaults to None.
    arfene_lo : numpy.ndarray, optional
        Lower edge of input model energy (ARF energy) bin. Defaults to None.
    arfene_hi : numpy.ndarray, optional
        Upper edge of input model energy (ARF energy) bin. Defaults to None.
    expo_lst : numpy.ndarray, optional
        Exposure list. Defaults to None
    int_rng : tuple of (float,float)
        The energy (keV) range for computing flux. Defaults to (1.0,2.3).
    arfscal_method : str
        Method for calculating ARFSCAL. Available methods are:
        - `FLX`: assuming all sources have same flux (erg/s/cm^2)
        - `LMN`: assuming all sources have same luminosity (erg/s)
        - `SHP`: assuming all sources have same spectral shape
    fits_name : str, optional
        If specified, create a fits file with name `fits_name`. Defaults to None.
    sample_arf : str, optional
        A sample ARF to read `arfene_lo` and `arfene_hi`. Defaults to sample.arf.
    srcid_lst : list or numpy.ndarray
        Source ID list. Defaults to None.
    prob_lst : list or numpy.ndarray
        A list of RMF 2D matrices. If given, the ARF used for calculating flux will be RMF-weighted. Defaults to None.

    Returns
    -------
    sum_specresp : numpy.ndarray
        The weighted sum of ARF profiles.
    '''
    
    specresp_lst = np.array(specresp_lst)
    pi_lst = np.array(pi_lst)
    z_lst = np.array(z_lst)

    if bkgpi_lst is None:
        bkgpi_lst = np.zeros_like(specresp_lst)
    if bkgscal_lst is None:
        bkgscal_lst = np.zeros(specresp_lst.shape[0])
    if ene_lo is None:
        ene = np.linspace(0,10,len(specresp_lst.shape[1])+1)
        ene_lo = ene[:-1]
    if ene_hi is None:
        ene = np.linspace(0,10,len(specresp_lst.shape[1])+1)
        ene_hi = ene[1:]
    if arfene_lo is None:
        arfene_lo = ene_lo
    if arfene_hi is None:
        arfene_hi = ene_hi
    if expo_lst is None:
        expo_lst = np.ones(specresp_lst.shape[1])
    if srcid_lst is None:
        srcid_lst = np.arange(len(specresp_lst))
    if prob_lst is None:
        prob_lst = np.full(len(srcid_lst),None)
    
    bkgpi_lst = np.array(bkgpi_lst)
    bkgscal_lst = np.array(bkgscal_lst)
    ene_lo = np.array(ene_lo)
    ene_hi = np.array(ene_hi)
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = ene_hi - ene_lo
    arfene_lo = np.array(arfene_lo)
    arfene_hi = np.array(arfene_hi)
    flg = (ene_ce > int_rng[0]) & (ene_ce < int_rng[-1])
    expo_lst = np.array(expo_lst)

    #specresp_ali_lst = np.zeros_like(specresp_lst)
    specresp_ali_lst = [[] for _ in range(len(specresp_lst))]
    for i in range(len(specresp_ali_lst)):
        specresp_ali_lst[i] = align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp_lst[i],prob_lst[i])
    specresp_ali_lst = np.array(specresp_ali_lst)
    arfscal_lst,arfnorm = get_arfscal(specresp_ali_lst,pi_lst,z_lst,bkgpi_lst,bkgscal_lst,expo_lst,ene_wd,flg,arfscal_method)
    specresp_scal_lst = specresp_lst * arfscal_lst[:,np.newaxis]
    sum_specresp = np.sum(specresp_scal_lst,axis=0)
    
    if fits_name is not None:
        with fits.open(sample_arf) as hdu:
            arf = hdu['SPECRESP'].data
        arfene_lo = arf['ENERG_LO']
        arfene_hi = arf['ENERG_HI']
        
        hdulist = fits.HDUList()
    
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        cols = [fits.Column(name='ENERG_LO', format='D', array=arfene_lo),
                fits.Column(name='ENERG_HI', format='D', array=arfene_hi),
                fits.Column(name='SPECRESP', format='D', array=sum_specresp)]
        hdu_specresp = fits.BinTableHDU.from_columns(cols, name='SPECRESP')
        # ARF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
        hdu_specresp.header['TELESCOP'] = 'STACKED'
        hdu_specresp.header['INSTRUME'] = 'STACKED'
        hdu_specresp.header['CHANTYPE'] = 'PI'
        hdu_specresp.header['DETCHANS'] = len(ene_ce)
        hdu_specresp.header['HDUCLASS'] = 'OGIP'
        hdu_specresp.header['HDUCLAS1'] = 'RESPONSE'
        hdu_specresp.header['HDUCLAS2'] = 'SPECRESP'
        hdu_specresp.header['HDUVERS'] = '1.1.0'
        hdu_specresp.header['EXPOSURE'] = expo_lst.sum()
        hdu_specresp.header['ARFMETH'] = arfscal_method
        hdu_specresp.header['CREATOR'] = 'XSTACK'
        hdulist.append(hdu_specresp)

        cols = [fits.Column(name='SRCID', format='J', array=srcid_lst),
                fits.Column(name='ARFSCAL', format='D', array=arfscal_lst),
                fits.Column(name='PHOCOUN', format='J', array=np.sum(pi_lst,axis=1)),
                fits.Column(name='BPHOCOUN', format='D', array=np.sum(bkgpi_lst*bkgscal_lst[:,np.newaxis],axis=1))]
        hdu_arfscal = fits.BinTableHDU.from_columns(cols, name='ARFSCAL')
        hdu_arfscal.header['ARFNORM'] = arfnorm
        hdulist.append(hdu_arfscal)

        cols = [fits.Column(name='CHANNEL', format='J', array=np.arange(1,len(flg)+1)),
                fits.Column(name='FLAG', format='J', array=flg.astype('int'))]
        hdu_flag = fits.BinTableHDU.from_columns(cols, name='FLAG')
        hdu_flag.header['FLAG'] = 'whether the bin is used for ARFSCAL estimation'
        hdulist.append(hdu_flag)
        
        hdulist.writeto('%s'%(fits_name), overwrite=True)
        
    return sum_specresp


def corr_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh):
    '''
    Multiply the ARF specresp with the galactic absorption profile. 
    The template galactic absorption profile should be at nh=1e20.
    The source nh value is specified by `nh`.

    Parameters
    ----------
    specresp : numpy.ndarray
        The ARF specresp to be corrected.
    arfene_lo : numpy.ndarray
        Lower edge of input model energy (ARF energy) bin.
    arfene_hi : numpy.ndarray
        Upper edge of input model energy (ARF energy) bin. Defaults to None.
    factor : numpy.ndarray
        Template galactic absorption profile at nh=1e20.
    nhene_lo : numpy.ndarray
        Lower edge of nh model energy bin.
    nhene_hi : numpy.ndarray
        Upper edge of nh model energy bin.
    nh : float
        Galactic nh of the source.

    Returns
    -------
    specresp_cor : numpy.ndarray
        The corrected ARF specresp.
    '''
    nhene_ce = (nhene_lo + nhene_hi) / 2
    nhene_wd = nhene_hi - nhene_lo
    nh_scal = nh / 1e20
    factor_scal = factor ** nh_scal
    specresp_cor = specresp.copy()
    # for each arf energy bin, find the nearest nh energy bins, and assign the correction factor
    # if more than one nh bins can be found, do interpolation
    for i in range(len(specresp_cor)):
        mask = (nhene_hi >= arfene_lo[i]) & (nhene_lo <= arfene_hi[i])
        if np.all(mask==False):
            continue
        nhene_mask_lo = nhene_lo[mask].copy()
        nhene_mask_hi = nhene_hi[mask].copy()
        nhene_mask_ce = nhene_ce[mask].copy()
        nhene_mask_wd = nhene_wd[mask].copy()
        factor_scal_mask = factor_scal[mask].copy()

        # for the first and last channel in the basket, we need to recalculate their widths
        nhene_mask_wd[0] = nhene_mask_hi[0] - arfene_lo[i]
        nhene_mask_wd[-1] = arfene_hi[i] - nhene_mask_lo[-1]
        
        prob_mask = nhene_mask_wd / nhene_mask_wd.sum()
        specresp_cor[i] *= (factor_scal_mask * prob_mask).sum()

    return specresp_cor


def get_arfscal(specresp_lst,pi_lst,z_lst,bkgpi_lst,bkgscal_lst,expo_lst,ene_wd,flg,method):
    '''
    Get the weighting factor for each ARF in a list.

    Parameters
    ----------
    specresp_lst : list or numpy.ndarray
        The ARF specresp (cm^2 vs. arf energy) list.
    pi_lst : list or numpy.ndarray
        PI spectrum list.
    z_lst : list or numpy.ndarray
        The redshift list.
    bkgpi_lst : list or numpy.ndarray
        Background PI spectrum list.
    bkgscal_lst : list or numpy.ndarray
        Background scaling-ratio list.
    expo_lst : list or numpy.ndarray
        Exposure list.
    ene_wd : numpy.ndarray
        Output channel energy bin width.
    flg : numpy.ndarray
        Output channel energy flag.
    method : str
        Method for calculating ARFSCAL. Available methods are:
        - `FLX`: assuming all sources have same flux (erg/s/cm^2)
        - `LMN`: assuming all sources have same luminosity (erg/s)
        - `SHP`: assuming all sources have same spectral shape

    Returns
    -------
    arfscal_lst : numpy.ndarray
        The ARF scaling ratio for each source.
    arfnorm : float
        The ARF weighting factor normalization.
    '''

    if method == 'FLX':     # FLUX
        arfscal_lst = expo_lst
        #arfscal_lst = expo_lst / expo_lst.sum()
        print(arfscal_lst)
        return arfscal_lst,1

    elif method == 'LMN':   # LUMINOSITY
        dist_lst = Planck18.luminosity_distance(z_lst).to(u.cm).value    # unit: Mpc
        arfscal_lst = expo_lst / dist_lst ** 2     # scaling ratio for each source
        #arfscal_lst = arfscal_lst / arfscal_lst.sum()
        print(arfscal_lst)
        return arfscal_lst,1

    elif method == 'SHP':   # SHAPE
        net_pi_lst = pi_lst - bkgpi_lst*bkgscal_lst[:,np.newaxis]
        net_pi_lst = net_pi_lst[:,flg]
        sum_net_pi_lst = np.sum(net_pi_lst,axis=1)

        resp_ene_lst = specresp_lst * ene_wd
        resp_ene_lst = resp_ene_lst[:,flg]
        sum_resp_ene_lst = np.sum(resp_ene_lst,axis=1)

        arfscal_lst = sum_net_pi_lst / sum_resp_ene_lst
        arfnorm = np.sum(arfscal_lst)
        arfscal_lst = arfscal_lst / np.sum(arfscal_lst)

        print(arfscal_lst)
        return arfscal_lst,arfnorm

    else:
        raise Exception('Available method for ARF scaling ratio calculation: `FLX`, `LMN`, or `SHP` !')
    
    return

        

def align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp,prob=None):
    '''
    The ARF energy bin and RMF energy bin (also the PI channel energy bin) does not always match. Align the ARF
    to get the effective area at each RMF energy bin.

    Parameters
    ----------
    ene_lo : numpy.ndarray
        Lower edge of output channel energy bin.
    ene_hi : numpy.ndarray
        Upper edge of output channel energy bin.
    arfene_lo : numpy.ndarray
        Lower edge of input model energy (ARF energy) bin.
    arfene_hi : numpy.ndarray
        Upper edge of input model energy (ARF energy) bin.
    specresp : numpy.ndarray
        The ARF specresp (cm^2 vs. arf energy).
    prob : numpy.ndarray
        RMF 2D matrix (prob.shape=(len(arfene_lo),len(ene_lo))).

    Returns
    -------
    specresp_ali : numpy.ndarray
        The aligned ARF specresp.
    '''
    assert ene_lo.shape == ene_hi.shape, ''
    
    if prob is None:
        arfene_wd = arfene_hi - arfene_lo
        specresp_ali = np.zeros(len(ene_lo))    # aligned specresp
        for i in range(len(specresp_ali)):
            mask = (ene_lo[i] <= arfene_hi) & (ene_hi[i] >= arfene_lo)
            if np.all(mask==False):
                continue
            arfene_mask_lo = arfene_lo[mask].copy()
            arfene_mask_hi = arfene_hi[mask].copy()
            arfene_mask_wd = arfene_wd[mask].copy()
            specresp_mask = specresp[mask].copy()
            
            # for the first and last masked channel, we need to recalculate their widths
            arfene_mask_wd[0] = arfene_mask_hi[0] - ene_lo[i]
            arfene_mask_wd[-1] = ene_hi[i] - arfene_mask_lo[-1]
            
            prob_mask = arfene_mask_wd / arfene_mask_wd.sum()
            specresp_ali[i] = (specresp_mask * prob_mask).sum()

    else:
        arfene_ce = (arfene_lo + arfene_hi) / 2
        arfene_wd = arfene_hi - arfene_lo
        ene_ce = (ene_lo + ene_hi) / 2
        ene_wd = ene_hi - ene_lo
        assert prob.shape[0] == len(arfene_ce), ''
        assert prob.shape[1] == len(ene_ce), ''

        specresp_arfenewd = specresp * arfene_wd
        specresp_arfenewd_ali = np.sum(specresp_arfenewd[:,np.newaxis]*prob,axis=0)
        specresp_ali = specresp_arfenewd_ali / ene_wd
        
    return specresp_ali