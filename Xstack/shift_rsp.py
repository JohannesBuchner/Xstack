import numpy as np
from astropy.io import fits
from numba import jit
from astropy.cosmology import Planck18



# shift_rsp
def shift_rsp(arffile,rmffile,z,nh_file=None,nh=1e20,ene_trc=None,rmfsft_method="NONPAR"):
    """
    Shift the ARF&RMF. This is literally done by three steps: 
    1) Combine input ARF and RMF into a single RSP matrix;
    1) Shift in the direction of output channel energy. That is to say, shift and broaden the probability profile for 
       each input energy (i.e. when the detector receive a photon with some input energy, the probability that a signal 
       at some output channel energy will be observed; so this is a function of output channel energy) by (1+z); 
    2) Shift in the direction of input energy by (1+z).
    
    Parameters
    ----------
    mat : astropy.io.fits.FITS_rec
        The `MATRIX` HDU data from a standard OGIP RMF file.
    ebo : astropy.io.fits.FITS_rec
        The `EBOUNDS` HDU data from a standard OGIP RMF file.
    z : float
        Redshift.
    rmfsft_method : str, optional
        The RMF shifting method. Two methods are available:
        - `PAR`: Parameterized method, i.e. approximate the probability profile with a Gaussian, and shift the Gaussians.
        - `NONPAR`: Non-PARameterized method, i.e. shift the probability profile directly. This should be more accurate, 
          and takes into account the off-diagonal elements in the RMF matrix. However, the non-PARameterized method is 
          more time-consuming than PARameterized method (~10^2 times slower).
          
    Returns
    -------
    prob_sft : numpy.ndarray
        The shifted probability matrix.
    """
    # read ARF and RMF file
    with fits.open(arffile) as hdu:
        arf = hdu["SPECRESP"].data    # SPECRESP extension
    arfene_lo = arf["ENERG_LO"].astype(np.float32)  # because @jit method do not accept >f4
    arfene_hi = arf["ENERG_HI"].astype(np.float32)
    arfene_ce = (arfene_lo + arfene_hi) / 2
    arfene_wd = arfene_hi - arfene_lo
    specresp = arf["SPECRESP"]

    with fits.open(rmffile) as hdu:
        mat = hdu["MATRIX"].data
        ebo = hdu["EBOUNDS"].data
    ene_lo = ebo["E_MIN"].astype(np.float32)
    ene_hi = ebo["E_MAX"].astype(np.float32)
    iene_lo = mat["ENERG_LO"].astype(np.float32)
    iene_hi = mat["ENERG_HI"].astype(np.float32)

    # sanity check: if the energy bins match
    assert np.all(arfene_lo==iene_lo), ""
    assert np.all(arfene_hi==iene_hi), ""

    # GalNH correction on ARF (optional)
    if nh_file is not None:
        with open(nh_file,"r") as file:
            lines = file.readlines()
        nhene_ce = []
        nhene_wd = []
        factor = []
        for line in lines:
            nhene_ce.append(float(line.split(" ")[0]))
            nhene_wd.append(float(line.split(" ")[1]))
            factor.append(float(line.split(" ")[2]))
        nhene_ce = np.array(nhene_ce)
        nhene_wd = np.array(nhene_wd)
        nhene_lo = nhene_ce - nhene_wd
        nhene_hi = nhene_ce + nhene_wd
        factor = np.array(factor)
        specresp = corr_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh)

    # truncate below ene_trc (optional)
    if ene_trc is not None:
        idx_trc = np.argmin(abs(arfene_ce-ene_trc))
        specresp[:idx_trc] = 0

    # combine ARF and RMF into a single RSP matrix
    prob = get_prob(mat,ebo)                # the RMF 2D matrix, shape=(iene_ce, ene_ce)
    rspmat = prob*specresp[:,np.newaxis]    # the RSP matrix (RMF*ARF)
    # rspmat,arfene_lo,arfene_hi,ene_lo,ene_hi = combine_arf_rmf(specresp,rmffile)

    # finally, shift the RSP matrix
    if rmfsft_method == "NONPAR":
        rspmat_sft = shift_matrix_nonpar(rspmat,arfene_lo,arfene_hi,ene_lo,ene_hi,z)
    elif rmfsft_method == "PAR":
        # TODO: add the parameterized method!
        raise Exception("NOT READY YET!")
    else:
        raise Exception('Available rmfsft_method (RMF shifting method): `PAR` or `NONPAR` (see help(shift_rmf) for illustration)!')   

    del mat,ebo 

    return rspmat_sft


# add_rsp
def add_rsp(rspmat_lst,pi_lst,z_lst,bkgpi_lst=None,bkgscal_lst=None,ene_lo=None,ene_hi=None,arfene_lo=None,arfene_hi=None,
            expo_lst=None,int_rng=(1.0,2.3),rspwt_method="SHP",outarf_name=None,sample_arf="sample.arf",srcid_lst=None,outrmf_name=None,sample_rmf="sample.rmf"):
    """
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
    """
    rspmat_lst = np.array(rspmat_lst)
    pi_lst = np.array(pi_lst)
    z_lst = np.array(z_lst)

    if bkgpi_lst is None:
        bkgpi_lst = np.zeros_like(rspmat_lst)
    if bkgscal_lst is None:
        bkgscal_lst = np.zeros(rspmat_lst.shape[0])
    if ene_lo is None:
        ene = np.linspace(0,10,len(rspmat_lst.shape[1])+1)
        ene_lo = ene[:-1]
    if ene_hi is None:
        ene = np.linspace(0,10,len(rspmat_lst.shape[1])+1)
        ene_hi = ene[1:]
    if arfene_lo is None:
        arfene_lo = ene_lo
    if arfene_hi is None:
        arfene_hi = ene_hi
    if expo_lst is None:
        expo_lst = np.ones(rspmat_lst.shape[1])
    if srcid_lst is None:
        srcid_lst = np.arange(len(rspmat_lst))
    
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

    rsp1d_lst = [[] for _ in range(len(rspmat_lst))]    # 1d specresp profile = 2d rsp matrix projected on output channel energy axis
    for i in range(len(rsp1d_lst)):
        rsp1d_lst[i] = project_rspmat(rspmat_lst[i],ene_lo,ene_hi,arfene_lo,arfene_hi)
    
    rsp1d_lst = np.array(rsp1d_lst)
    rspwt_lst,rspnorm = compute_rspwt(rsp1d_lst,pi_lst,z_lst,bkgpi_lst,bkgscal_lst,expo_lst,ene_wd,flg,rspwt_method)

    rspmat_wt_lst = rspmat_lst * rspwt_lst[:,np.newaxis,np.newaxis]
    sum_rspmat = np.sum(rspmat_wt_lst,axis=0)

    # extract ARF
    sum_specresp = np.sum(sum_rspmat,axis=1)

    if outarf_name is not None:
        with fits.open(sample_arf) as hdu:
            arf = hdu["SPECRESP"].data
        arfene_lo = arf["ENERG_LO"]
        arfene_hi = arf["ENERG_HI"]
        
        hdulist = fits.HDUList()
    
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        cols = [fits.Column(name="ENERG_LO", format="D", array=arfene_lo),
                fits.Column(name="ENERG_HI", format="D", array=arfene_hi),
                fits.Column(name="SPECRESP", format="D", array=sum_specresp)]
        hdu_specresp = fits.BinTableHDU.from_columns(cols, name="SPECRESP")
        # ARF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
        hdu_specresp.header["TELESCOP"] = "STACKED"
        hdu_specresp.header["INSTRUME"] = "STACKED"
        hdu_specresp.header["CHANTYPE"] = "PI"
        hdu_specresp.header["DETCHANS"] = len(ene_ce)
        hdu_specresp.header["HDUCLASS"] = "OGIP"
        hdu_specresp.header["HDUCLAS1"] = "RESPONSE"
        hdu_specresp.header["HDUCLAS2"] = "SPECRESP"
        hdu_specresp.header["HDUVERS"] = "1.1.0"
        hdu_specresp.header["EXPOSURE"] = expo_lst.sum()
        hdu_specresp.header["ARFMETH"] = rspwt_method
        hdu_specresp.header["CREATOR"] = "XSTACK"
        hdulist.append(hdu_specresp)

        cols = [fits.Column(name="SRCID", format="J", array=srcid_lst),
                fits.Column(name="RSPWT", format="D", array=rspwt_lst),
                fits.Column(name="PHOCOUN", format="J", array=np.sum(pi_lst,axis=1)),
                fits.Column(name="BPHOCOUN", format="D", array=np.sum(bkgpi_lst*bkgscal_lst[:,np.newaxis],axis=1))]
        hdu_arfscal = fits.BinTableHDU.from_columns(cols, name="ARFSCAL")
        hdu_arfscal.header['RSPNORM'] = rspnorm
        hdulist.append(hdu_arfscal)

        cols = [fits.Column(name="CHANNEL", format="J", array=np.arange(1,len(flg)+1)),
                fits.Column(name="FLAG", format="J", array=flg.astype("int"))]
        hdu_flag = fits.BinTableHDU.from_columns(cols, name='FLAG')
        hdu_flag.header["FLAG"] = "whether the bin is used for ARFSCAL estimation"
        hdulist.append(hdu_flag)
        
        hdulist.writeto(f"{outarf_name}", overwrite=True)

    # extract RMF
    sum_prob = sum_rspmat / sum_specresp[:,np.newaxis]

    if outrmf_name is not None:
        hdulist = fits.HDUList()
        
        # extension 0: primary hdu
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        # extension 1: MATRIX
        with fits.open(sample_rmf) as hdu:
            mat = hdu['MATRIX'].data
            ebo = hdu['EBOUNDS'].data
        iene_lo = mat['ENERG_LO']
        iene_hi = mat['ENERG_HI']
        n_grp = []
        f_chan = []
        n_chan = []
        mat = []
        for i in range(len(iene_lo)):
            n_grp.append(1)
            f_chan.append(np.array([1]))
            sum_prob_i = sum_prob[i]
            # Find the index of the first non-zero element from the end
            last_nonzero_idx = len(sum_prob_i) - np.argmax(sum_prob_i[::-1] != 0) - 1
            n_chan.append(np.array([last_nonzero_idx+1]))
            mat.append(sum_prob_i[:last_nonzero_idx+1])
        n_grp = np.array(n_grp)
            
        cols = [fits.Column(name='ENERG_LO', format='D', array=iene_lo),
                fits.Column(name='ENERG_HI', format='D', array=iene_hi),
                fits.Column(name='N_GRP', format='J', array=n_grp),
                fits.Column(name='F_CHAN', format='PJ()', array=f_chan),
                fits.Column(name='N_CHAN', format='PJ()', array=n_chan),
                fits.Column(name='MATRIX', format='PD()', array=mat)]
        hdu_matrix = fits.BinTableHDU.from_columns(cols, name='MATRIX')
        # RMF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
        hdu_matrix.header['TELESCOP'] = 'STACKED'
        hdu_matrix.header['INSTRUME'] = 'STACKED'
        hdu_matrix.header['CHANTYPE'] = 'PI'
        hdu_matrix.header['DETCHANS'] = sum_prob.shape[1]
        hdu_matrix.header['HDUCLASS'] = 'OGIP'
        hdu_matrix.header['HDUCLAS1'] = 'RESPONSE'
        hdu_matrix.header['HDUCLAS2'] = 'RSP_MATRIX'
        hdu_matrix.header['HDUVERS'] = '1.3.0'
        hdu_matrix.header['TLMIN4'] = 1 # the first channel in the response
        hdu_matrix.header['EXPOSURE'] = expo_lst.sum()
        hdu_matrix.header['ANCRFILE'] = outarf_name # TODO: save under the same dir?
        hdu_matrix.header['CREATOR'] = 'XSTACK'
        hdulist.append(hdu_matrix)
        
        # extension 2: EBOUNDS
        ene_lo = ebo['E_MIN']
        ene_hi = ebo['E_MAX']
        chan = np.arange(1,len(ene_lo)+1)
        cols = [fits.Column(name='CHANNEL', format='J', array=chan),
                fits.Column(name='E_MIN', format='D', array=ene_lo),
                fits.Column(name='E_MAX', format='D', array=ene_hi)]
        hdu_ebounds = fits.BinTableHDU.from_columns(cols, name='EBOUNDS')
        # RMF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
        hdu_ebounds.header['TELESCOP'] = 'STACKED'
        hdu_ebounds.header['INSTRUME'] = 'STACKED'
        hdu_ebounds.header['CHANTYPE'] = 'PI'
        hdu_ebounds.header['DETCHANS'] = sum_prob.shape[1]
        hdu_ebounds.header['HDUCLASS'] = 'OGIP'
        hdu_ebounds.header['HDUCLAS1'] = 'RESPONSE'
        hdu_ebounds.header['HDUCLAS2'] = 'EBOUNDS'
        hdu_ebounds.header['HDUVERS'] = '1.2.0'
        hdulist.append(hdu_ebounds)
        
        # extension 3: RMFSCAL
        cols = [fits.Column(name='SRCID', format='J', array=srcid_lst),
                fits.Column(name='RSPWT', format='D', array=rspwt_lst)]
        hdu_rmfscal = fits.BinTableHDU.from_columns(cols, name='RMFSCAL')
        hdulist.append(hdu_rmfscal)
        
        hdulist.writeto(f"{outrmf_name}", overwrite=True)

    return sum_specresp, sum_prob


#==============================================
########### Non-Par shifting func #############
#==============================================
@jit
def shift_matrix_nonpar(prob,iene_lo,iene_hi,ene_lo,ene_hi,z):
    '''
    Numba code for Non-parametric RSP/RMF shifting.

    Parameters
    ----------
    prob : numpy.ndarray
        The RSP/RMF 2D probability matrix, or the RSP 2D matrix.
    iene_lo : numpy.ndarray
        Lower edge of input model energy (ARF energy) bin.
    iene_hi : numpy.ndarray
        Upper edge of input model energy (ARF energy) bin.
    ene_lo : numpy.ndarray
        Lower edge of output channel energy bin.
    ene_hi : numpy.ndarray
        Upper edge of output channel energy bin.
    z : float
        Redshift.

    Returns
    -------
    prob_sft : numpy.ndarray
        The rest-frame shifted RSP/RMF 2D probability matrix. 
    '''
    iene_ce = (iene_lo + iene_hi) / 2
    iene_wd = iene_hi - iene_lo
    iene_id = np.arange(len(iene_ce))

    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = ene_hi - ene_lo
    ene_id = np.arange(len(ene_ce))
    
    # de-redshift probability matrix
    # step 1: horizontal shift, output channel energy *(1+z), dispersion automatically *(1+z)
    prob_sft_horizontal = np.zeros(prob.shape)  # the probability matrix after step 1: horizontal shift
    iene_ubound = np.max(iene_lo)
    iene_lbound = np.min(iene_hi)
    for i in range(len(iene_ce)):

        iene_lo_map = iene_lo[i] * (1+z)
        iene_hi_map = iene_hi[i] * (1+z)
        
        if iene_lo_map > iene_ubound:
            break
        if iene_hi_map < iene_lbound:
            continue

        prob_1d = np.zeros(len(ene_ce))
        ene_ubound = np.max(ene_lo)
        ene_lbound = np.min(ene_hi)
        for j in range(len(ene_ce)):
            ene_lo_map = ene_lo[j] * (1+z)
            ene_hi_map = ene_hi[j] * (1+z)
            
            if ene_lo_map > ene_ubound:
                break
            if ene_hi_map < ene_lbound:
                continue
            
            mask = (ene_lo_map < ene_hi) & (ene_hi_map > ene_lo)
            ene_id_mask = ene_id[mask]
            ene_wd_mask = ene_wd[mask]
            ene_lo_mask = ene_lo[mask]
            ene_hi_mask = ene_hi[mask]
            
            ene_wd_mask[0] = ene_hi_mask[0] - ene_lo_map
            ene_wd_mask[-1] = ene_hi_map - ene_lo_mask[-1]
            
            prob_mask = ene_wd_mask / np.sum(ene_wd_mask)
            
            prob_1d[ene_id_mask] += prob[i][j] * prob_mask
        
        if np.sum(prob_1d) > 0:
            prob_1d *= np.sum(prob[i])/np.sum(prob_1d)
        prob_sft_horizontal[i] = prob_1d
            
    # step 2: vertical shift, input model energy *(1+z)
    prob_sft_vertical = np.zeros(prob.shape)

    iene_sft_lo = iene_lo * (1+z)
    iene_sft_hi = iene_hi * (1+z)
    iene_sft_ce = iene_ce * (1+z)
    iene_sft_wd = iene_wd * (1+z)

    for i in range(prob_sft_vertical.shape[0]):
        mask = (iene_lo[i] <= iene_sft_hi) & (iene_hi[i] >= iene_sft_lo)
        if np.all(mask==False):
            continue
        iene_mask_lo = iene_sft_lo[mask].copy()
        iene_mask_hi = iene_sft_hi[mask].copy()
        iene_mask_ce = iene_sft_ce[mask].copy()
        iene_mask_wd = iene_sft_wd[mask].copy()
        prob_sft_horizontal_mask = prob_sft_horizontal[mask].copy()
        
        # for the first and last channel in the basket, we need to recalculate their widths
        iene_mask_wd[0] = iene_mask_hi[0] - iene_lo[i]
        iene_mask_wd[-1] = iene_hi[i] - iene_mask_lo[-1]
        
        prob_mask = iene_mask_wd / iene_mask_wd.sum()
        prob_sft_vertical[i] = np.sum(prob_sft_horizontal_mask*prob_mask[:,np.newaxis],axis=0)

    return prob_sft_vertical


def project_rspmat(rspmat,ene_lo,ene_hi,arfene_lo,arfene_hi,proj_axis="CHANNEL"):
    """
    Project the 2D RSP matrix onto CHANNEL/MODEL energy axis, to get the effective specresp (cm^2 vs. energy)
    
    Parameters
    ----------
    rspmat : numpy.ndarray
        The 2D RSP matrix.
    ene_lo : numpy.ndarray
        Lower edge of output channel energy bin.
    ene_hi : numpy.ndarray
        Upper edge of output channel energy bin.
    arfene_lo : numpy.ndarray
        Lower edge of input model energy (ARF energy) bin.
    arfene_hi : numpy.ndarray
        Upper edge of input model energy (ARF energy) bin.
    

    Returns
    -------
    specresp_ali : numpy.ndarray
        The aligned ARF specresp.
    """
    # sanity check
    assert ene_lo.shape == ene_hi.shape, ""
    assert arfene_lo.shape == arfene_hi.shape, ""
    assert rspmat.shape[0] == len(arfene_lo), ""
    assert rspmat.shape[1] == len(ene_lo), ""

    arfene_ce = (arfene_lo + arfene_hi) / 2
    arfene_wd = arfene_hi - arfene_lo
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = ene_hi - ene_lo

    if proj_axis == "CHANNEL":
        rspmat_arfenewd = rspmat*arfene_wd[:,np.newaxis]
        rsp1d = np.sum(rspmat_arfenewd,axis=0) / ene_wd

    elif proj_axis == "MODEL":
        rsp1d = np.sum(rspmat,axis=1)

    else:
        raise Exception("Invalid `proj_axis` parameter (available: `CHANNEL` or `MODEL`)!")

    return rsp1d


def compute_rspwt(specresp_lst,pi_lst,z_lst,bkgpi_lst,bkgscal_lst,expo_lst,ene_wd,flg,method):
    """
    Get the weighting factor for each RSP in a list.

    Parameters
    ----------
    specresp_lst : list or numpy.ndarray
        The list of RSP specresp projected on channel energy axis (cm^2 vs. channel energy).
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
        - `SHP`: assuming all sources have same spectral shape
        - `FLX`: assuming all sources have same flux (erg/s/cm^2)
        - `LMN`: assuming all sources have same luminosity (erg/s)

    Returns
    -------
    arfscal_lst : numpy.ndarray
        The ARF scaling ratio for each source.
    arfnorm : float
        The ARF weighting factor normalization.
    """

    if method == "SHP":   # SHAPE; This is the minimum assumption for spectral stacking, that all spectra look similar in shape
        net_pi_lst = pi_lst - bkgpi_lst*bkgscal_lst[:,np.newaxis]
        net_pi_lst = net_pi_lst[:,flg]
        sum_net_pi_lst = np.sum(net_pi_lst,axis=1)

        resp_ene_lst = specresp_lst * ene_wd
        resp_ene_lst = resp_ene_lst[:,flg]
        sum_resp_ene_lst = np.sum(resp_ene_lst,axis=1)

        rspwt_lst = sum_net_pi_lst / sum_resp_ene_lst
        rspnorm = np.sum(rspwt_lst)
        rspwt_lst = rspwt_lst / np.sum(rspwt_lst)

    elif method == "FLX":     # FLUX
        rspwt_lst = expo_lst
        rspnorm = 1

    elif method == "LMN":   # LUMINOSITY
        dist_lst = Planck18.luminosity_distance(z_lst).to(u.cm).value    # unit: Mpc
        rspwt_lst = expo_lst / dist_lst ** 2     # scaling ratio for each source
        rspnorm = 1     # arbitrary number

    else:
        raise Exception('Available method for ARF scaling ratio calculation: `FLX`, `LMN`, or `SHP` !')
    
    print(rspwt_lst)
    
    return rspwt_lst,rspnorm


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


def get_prob(mat,ebo):
    '''
    Parse the RMF file (input the `MATRIX` and `EBOUNDS` extension) into a 2D probability matrix. 

    Parameters
    ----------
    mat : astropy.io.fits.FITS_rec
        The `MATRIX` extension of a standard OGIP RMF file. Must include the following columns:
        - `ENERG_LO`
        - `ENERG_HI`
        - `N_GRP`
        - `F_CHAN`
        - `N_CHAN`
        - `MATRIX`
    ebo : astropy.io.fits.FITS_rec
        The `EBOUNDS` extension of a standard OGIP RMF file. Must include the following columns:
        - `E_MIN` 
        - `E_MAX`

    Returns
    -------
    prob : numpy.ndarray
        The RMF 2D probability matrix. Index [i,j], where:
        - i represents arfene (iene or inpu model energy)
        - j represents ene (output channel energy)
    '''
    ene_lo = ebo['E_MIN'].astype(np.float32)
    ene_hi = ebo['E_MAX'].astype(np.float32)
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = ene_hi - ene_lo
    iene_lo = mat['ENERG_LO'].astype(np.float32)
    iene_hi = mat['ENERG_HI'].astype(np.float32)
    iene_ce = (iene_lo + iene_hi) / 2
    iene_wd = iene_hi - iene_lo
    grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
    prob = np.zeros(grid[0].shape) # probability per channel
    
    n_grp = mat['N_GRP']
    f_chan = mat['F_CHAN']
    n_chan = mat['N_CHAN']
    matrix = np.array(mat['MATRIX'])
    
    f_chan_0 = int(np.min([np.min(f_chan[_]) for _ in range(len(f_chan))])) # the zero point of channel index
    for i in range(len(iene_ce)):
        f_matrix = 0   # starting index of matrix[i]
        for grp_j in range(n_grp[i]):
            f_chan_j = f_chan[i][grp_j] - f_chan_0  # starting index of channel
            n_chan_j = n_chan[i][grp_j]             # number of channel
            e_chan_j = f_chan_j + n_chan_j          # ending index of in channel
            e_matrix = f_matrix + n_chan_j          # ending index of matrix[i]
            
            prob[i][f_chan_j:e_chan_j] += matrix[i][f_matrix:e_matrix]
            f_matrix += n_chan_j

    return prob