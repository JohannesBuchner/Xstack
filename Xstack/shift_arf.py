import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18
import astropy.units as u
from tqdm import tqdm


def shift_arf(arf_file,z,nh_file=None,nh=1e20):
    '''
    Shift ARF in the same way as the PHA file. Galactic absorption correction will be applied on the ARF before shifting,
    if `nh_file` is specified.

    In the shifted ARF:
    The lower energy edges and higher energy edges remain the same as the unshifted ARF, but specresp is taken as the 
    value that corresponds to the nearest shifted energy bin.
    e.g. specresp_sft@2keV = specresp@1keV, if redshift=1

    Parameters
    ----------
    arf_file : str
        The name of arf_file.
    z : float
        Redshift of the source.
    nh_file : str, optional
        Galactic absorption profile (absorption factor vs. energy). If specified, galactic absorption 
        correction will be applied on the ARF before shifting.
        * Should be in txt format. 
        * Should also contain the following columns in the first extension: `nhene_ce`, `nhene_wd`, `factor`.
        * `factor` should indicate the absorption factor when nh=1e20.
        * An easy way to obtain the `nh_file`: iplot `tbabs*powerlaw` with `Nh`=1e20 and `PhoIndex`=0.0, `Norm`=1 in Xspec.
    nh : float
        The galactic absorption nh of the source. (e.g. 3e20)

    Returns
    -------
    specresp_sft : ndarray
        The shifted (and also galactic absorption corrected) ARF profile (cm^2 vs. arf energy).
    '''

    with fits.open(arf_file) as hdu:
        arf = hdu['SPECRESP'].data
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
        nhene_lo = nhene_ce - 0.5 * nhene_wd
        nhene_hi = nhene_ce + 0.5 * nhene_wd
        factor = np.array(factor)
        # with fits.open(nh_file) as hdu:
        #     phabs = hdu[1].data
        # nhene_ce = phabs['nhene_ce']
        # nhene_wd = phabs['nhene_wd']
        # nhene_lo = nhene_ce - 0.5 * nhene_wd
        # nhene_hi = nhene_ce + 0.5 * nhene_wd
        # factor = phabs['factor']
        specresp = corr_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh)
    
    arfene_sft_lo = arfene_lo * (1+z)
    arfene_sft_hi = arfene_hi * (1+z)
    arfene_sft_ce = arfene_ce * (1+z)
    arfene_sft_wd = arfene_wd * (1+z)
    
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


def add_arf(specresp_lst,pha_lst,z_lst,bkgpha_lst=None,bkgscal_lst=None,ene_lo=None,ene_hi=None,arfene_lo=None,arfene_hi=None,
            expo_lst=None,int_rng=(1.0,2.3),arfscal_method='SHP',fits_name=None,sample_arf='sample.arf',srcid_lst=None):
    '''
    Weighted sum of ARF profiles.

    Parameters
    ----------
    specresp_lst : list or array_like
        The ARF profiles (cm^2 vs. arf energy) list.
    pha_lst : list or array_like
        The PHA list.
    z_lst : list or array_like
        The redshift list.

    Returns
    -------
    sum_specresp : ndarray
        The weighted sum of ARF profiles.
    '''
    
    specresp_lst = np.array(specresp_lst)
    pha_lst = np.array(pha_lst)
    z_lst = np.array(z_lst)

    if bkgpha_lst is None:
        bkgpha_lst = np.zeros_like(specresp_lst)
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
    
    bkgpha_lst = np.array(bkgpha_lst)
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
        specresp_ali_lst[i] = align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp_lst[i])
    specresp_ali_lst = np.array(specresp_ali_lst)
    arfscal_lst = get_arfscal(specresp_ali_lst,pha_lst,z_lst,bkgpha_lst,bkgscal_lst,expo_lst,ene_wd,flg,arfscal_method)
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
        hdu_spectrum = fits.BinTableHDU.from_columns(cols, name='SPECRESP')
        # ARF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
        hdu_spectrum.header['TELESCOP'] = 'STACKED'
        hdu_spectrum.header['INSTRUME'] = 'STACKED'
        hdu_spectrum.header['CHANTYPE'] = 'PI'
        hdu_spectrum.header['DETCHANS'] = len(ene_ce)
        hdu_spectrum.header['HDUCLASS'] = 'OGIP'
        hdu_spectrum.header['HDUCLAS1'] = 'RESPONSE'
        hdu_spectrum.header['HDUCLAS2'] = 'SPECRESP'
        hdu_spectrum.header['HDUVERS'] = '1.1.0'
        hdu_spectrum.header['EXPOSURE'] = expo_lst.sum()
        hdu_spectrum.header['ARFMETH'] = arfscal_method
        hdu_spectrum.header['CREATOR'] = 'XSTACK'
        hdulist.append(hdu_spectrum)

        cols = [fits.Column(name='SRCID', format='J', array=srcid_lst),
                fits.Column(name='ARFSCAL', format='D', array=arfscal_lst),
                fits.Column(name='PHOCOUN', format='J', array=np.sum(pha_lst,axis=1)),
                fits.Column(name='BPHOCOUN', format='D', array=np.sum(bkgpha_lst*bkgscal_lst[:,np.newaxis],axis=1))]
        hdu_arfscal = fits.BinTableHDU.from_columns(cols, name='ARFSCAL')
        hdulist.append(hdu_arfscal)

        cols = [fits.Column(name='CHANNEL', format='J', array=np.arange(1,len(flg)+1)),
                fits.Column(name='FLAG', format='J', array=flg.astype('int'))]
        hdu_flag = fits.BinTableHDU.from_columns(cols, name='FLAG')
        hdu_flag.header['FLAG'] = 'whether the bin is used for ARFSCAL estimation'
        hdulist.append(hdu_flag)
        
        hdulist.writeto(fits_name, overwrite=True)
        
    return sum_specresp


def corr_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh):
    '''
    Multiply the ARF profile with the galactic absorption profile. 
    The template galactic absorption profile should be at nh=1e20.
    The source nh value is specified by `nh`.

    Parameters
    ----------
    specresp : array_like
        The ARF profile to be corrected.
    arfene_lo : array_like
        Lower edge of ARF energy bin.
    arfene_hi : array_like
        Upper edge of ARF energy bin.
    factor : array_like
        Template galactic absorption profile at nh=1e20.
    nhene_lo : array_like
        Lower edge of nh energy bin.
    nhene_hi : array_like
        Upper edge of ARF energy bin.
    nh : float
        Galactic nh of the source.

    Returns
    -------
    specresp_cor : array_like
        The corrected ARF profile.
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


def get_arfscal(specresp_lst,pha_lst,z_lst,bkgpha_lst,bkgscal_lst,expo_lst,ene_wd,flg,method):
    '''
    Get the weighting factor for each ARF in a list.

    Parameters
    ----------
    specresp_lst : list or array_like
        ARF profile list (effective area at each ARF energy bin).
    pha_lst : list or array_like
        PHA file list.
    z_lst : list or array_like
        The redshift list.
    bkgpha_lst : list or array_like
        Background PHA file list.
    bkgscal_lst : list or array_like
        Background scaling ratio list.
    expo_lst : list or array_like
        Exposure list.
    ene_wd : array_like
        Energy bin width.
    flg : array_like
        Energy flag.
    method : str
        Method for calculating ARFSCAL. Available methods are:
        * `FLX`: assuming all sources have same flux (erg/s/cm^2)
        * `LMN`: assuming all sources have same luminosity (erg/s)
        * `SHP`: assuming all sources have same spectral shape

    Returns
    -------
    arfscal_lst : ndarray
        The ARF scaling ratio for each source.
    '''

    if method == 'FLX':     # FLUX
        arfscal_lst = expo_lst
        #arfscal_lst = expo_lst / expo_lst.sum()
        print(arfscal_lst)
        return arfscal_lst

    elif method == 'LMN':   # LUMINOSITY
        dist_lst = Planck18.luminosity_distance(z_lst).to(u.cm).value    # unit: Mpc
        arfscal_lst = expo_lst / dist_lst ** 2     # scaling ratio for each source
        #arfscal_lst = arfscal_lst / arfscal_lst.sum()
        print(arfscal_lst)
        return arfscal_lst

    elif method == 'SHP':   # SHAPE
        net_pha_lst = pha_lst - bkgpha_lst*bkgscal_lst[:,np.newaxis]
        net_pha_lst = net_pha_lst[:,flg]
        sum_net_pha_lst = np.sum(net_pha_lst,axis=1)

        resp_ene_lst = specresp_lst * ene_wd
        resp_ene_lst = resp_ene_lst[:,flg]
        sum_resp_ene_lst = np.sum(resp_ene_lst,axis=1)

        arfscal_lst = sum_net_pha_lst / sum_resp_ene_lst
        arfscal_lst = arfscal_lst / np.sum(arfscal_lst)

        print(arfscal_lst)
        return arfscal_lst
    
        # # old version
        # pm_lst = (pha_lst - bkgpha_lst*bkgscal_lst[:,np.newaxis]) / specresp_lst / ene_wd    # predicted-model*expo list
        # pm_flg_lst = pm_lst[:,flg]
        # pm_flg_lst[np.isnan(pm_flg_lst)] = 0 # remove NaN
        # arfscal_lst = np.sum(pm_flg_lst,axis=1)
        # arfscal_lst = arfscal_lst / arfscal_lst.sum()

    else:
        raise Exception('Available method for ARF scaling ratio calculation: `FLX`, `LMN`, or `SHP` !')
    # # Old version
    # for i in tqdm(range(len(pm_lst))):
    #     pm = pm_lst[i]
    #     try:
    #         vi = np.isfinite(pm) & (ene_ce > 1.5) & (ene_ce < 8)
    #         params, covariance = curve_fit(po, ene_ce[vi], pm[vi])
    #         pm_lst[i] = po(ene_ce,*params)
    #         pcidx_lst.append(-params[1])
    #     except Exception as e:
    #         vi = np.isfinite(pm)
    #         interp_func = interp1d(ene_ce[vi], pm[vi], kind='linear', fill_value='extrapolate')
    #         pm_lst[i] = interp_func(ene_ce)
    #         print(i)
    #         badq += 1
    #         pcidx_lst.append(0)
    return

        

def align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp):
    '''
    The ARF energy bin and RMF energy bin (also the PHA channel energy bin) does not always match. Align the ARF
    to get the effective area at each RMF energy bin.

    Parameters
    ----------
    ene_lo : array_like
        Lower edge of RMF energy bin (also the PHA channel energy bin).
    ene_hi : array_like
        Upper edge of RMF energy bin (also the PHA channel energy bin).
    arfene_lo : array_like
        Lower edge of ARF energy bin.
    arfene_hi : array_like
        Upper edge of RMF energy bin.
    specresp : array_like
        ARF profile (effective area at each ARF energy bin).

    Returns
    -------
    specresp_ali : ndarray
        The aligned ARF profile.
    '''
    assert ene_lo.shape == ene_hi.shape, ''
    
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
        
    return specresp_ali
