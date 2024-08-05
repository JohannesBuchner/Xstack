##############################################
########### PHA SHIFT & STACK ################
##############################################
import numpy as np
from astropy.io import fits


def shift_pha(pha_file,rmf_file,z):
    '''
    Shift a single PHA to rest-frame.
    
    Parameters
    ----------
    pha_file: Observed-frame PHA file to be shifted.
    rmf_file: RMF file (OGIP format) that defines channel-energy conversion.
    z: redshift
    
    Returns
    -------
    rest_chan: Rest-frame channel.
    rest_coun: Photon counts in each rest-frame channel.
    pha_chan: Observed-frame channel.
    pha_coun: Photon counts in each observed-frame channel.
    '''
    #print('############Spectrum Shifting#############')
    #tstart = time.time()

    with fits.open(rmf_file) as hdu:
        rmf = hdu['MATRIX'].data # default: matrix stored in table 1
        ebo = hdu['EBOUNDS'].data # ebounds
    
    chan = ebo['CHANNEL']
    ene_lo = ebo['E_MIN']
    ene_hi = ebo['E_MAX']
    ene_ce = (ene_lo + ene_hi)/2
    ene_wd = ene_hi - ene_lo
    ene_id = np.arange(len(ene_ce))
    
    with fits.open(pha_file) as hdu:
        pi = hdu['SPECTRUM'].data
    pha_chan = pi['CHANNEL'] # pha_chan starts from 0
    pha_coun = pi['COUNTS'] # the obs-frame photon counts
    assert (pha_chan == chan).all()
    chan_id = np.arange(len(pha_chan))
    
    rest_chan = pha_chan.copy()
    rest_coun = np.zeros(len(rest_chan),dtype=int) # the src-frame photon counts
    
    ene_ubound = ene_lo.max()
    ene_lbound = ene_hi.min() # set lower and upper bound of energy to avoid overflow issues
    
    for i in range(len(pha_chan)):
        #print(i)
        ene_lo_map = ene_lo[i] * (1+z)
        ene_hi_map = ene_hi[i] * (1+z)
        if ene_lo_map > ene_ubound:
            continue
        if ene_hi_map < ene_lbound:
            continue
        
        mask = (ene_hi > ene_lo_map) & (ene_lo < ene_hi_map)
        ene_id_mask = ene_id[mask]
        ene_wd_mask = ene_wd[mask]
        ene_lo_mask = ene_lo[mask]
        ene_hi_mask = ene_hi[mask]
        
        chan_id_mask = chan_id[mask]
        
        # for the first and last channel in the basket, we need to recalculate their widths
        # this is because they are defined by chan_lo_map[i] and chan_hi_map[i], respectively
        ene_wd_mask[0] = ene_hi_mask[0] - ene_lo_map
        ene_wd_mask[-1] = ene_hi_map - ene_lo_mask[-1]
        
        # each channel in the basket would get number of photons proportional to its width
        prob_mask = ene_wd_mask / ene_wd_mask.sum() # the probability of entering each channel in the basket
        phoct_mask = (pha_coun[i] * prob_mask).astype(int)
        # if there are more than 1 channel in the basket, we want to make sure that the sum of photons in all bins are equal to pha_coun[i]
        if len(phoct_mask) > 1:
            phoct_mask[0] = pha_coun[i] - phoct_mask[1:].sum()
        
        # finally, assign the photons
        for idx in range(len(chan_id_mask)):
            try:
                rest_coun[chan_id_mask[idx]] += phoct_mask[idx]
            except IndexError:
                continue
                
    #print('Spectrum has been shifted (z=%.4f)\nTime used: %.2f s'%(z,time.time()-tstart))
    #print('------------------------------------------')

    return (rest_chan, rest_coun, pha_chan, pha_coun)


def add_pha(pha_lst,scal_lst=None,fits_name=None,expo=10,bkg_file=None,rmf_file=None,arf_file=None):
    '''
    The weighted sum of many PHA files. The weights are specified by `scal_lst`. 
    
    If source PHAs are to be summed, the weights should all be unity.
    Otherwise if background PHAs are to be summed, the weights should be (as a return from function `get_bkgscal`):
        `src_areascal / bkg_areascal * src_backscal / bkg_backscal * src_expo / bkg_expo`
    
    The uncertainty in each channel is calculated with Gaussian error propogation.
    
    Caution should be taken when adding background spectra with varing scaling ratios:
    1. If all backgrounds have same scaling ratio, we can simply add them together. 
    2. However if the scaling ratio for each spectrum varies, we need to scale them first before summing together. Since 
    the scaling ratio for background spectrum is often a number much smaller than 1, each scaled background spectrum 
    may have float number of photon counts < 1 in some channel i. In this case, the uncertainty in channel i cannot 
    be calculated with Poisson statistics (i.e. sqrt(N)). 
    3. To conclude, `add_bkgpha` should be used when considering error of stacked background spectra with varied scaling 
    ratio. `add_bkgpha` first group background spectra with similar scaling ratios, then calculate error for each group 
    with Poisson statistics (each channel has enough photon counts now), and finally calculate the error for the total 
    summed background spectra (each group of spectra is in high-counts regime, so Gaussian error propagation works).
    4. Nevertheless for the photon counts (rather than the error), it is still recommended to use `add_pha`.
    
    Explanations
    ------------
    The Net counts for some source:
        
        net counts = source counts - background counts * scaling factor                                           (1)
        
        where scaling factor = src_areascal / bkg_areascal * src_backscal / bkg_backscal * src_expo / bkg_expo,
        and should be a return from function `get_bkgscal`.
        
    The error propagation tells us that:
    
        Err(net)**2 = Err(source)**2 + Err(background)**2 * scaling factor**2                                     (2)
        
    Assuming Poisson distribution,
    
        Err(source)**2 = source counts                                                                            (3)
        Err(background)**2 = background counts                                                                    (4)
        
    So the error for source spectrum and **scaled** background spectrum to be used in background subtracting should be:
        
        Err(source,scaled) = sqrt(source counts)                                                                  (5)
        Err(background,scaled) = sqrt(background counts) * scaling factor                                         (6)
        
    Note that for the source spectrum, the scaling factor is by convention 1;
    so the two equations have same mathematical form.
    
    Also note that the additional factor for the error is **scaling factor**, not **sqrt(scaling factor)**. This should
    be reasonable:
    
    1. `sqrt(background counts) / sqrt(scaling factor)` tells us the error of background spectrum, when it is estimated 
    from a region as small as the source region. 
    
    2. However, we often estimate the background from a region much larger than the source region (determined by scaling 
    factor): so we should have better understanding of the background spectrum, and therefore smaller uncertainties, as 
    expected from `sqrt(Background counts) / scaling factor`.
    
    Parameters
    ----------
    pha_lst: PHA file list.
    scal_lst: Weight (scaling ratio) list.
    fits_name: If specified, create a fits file with name `fits_name`. (Default is None)
    
    Returns
    -------
    sum_pha, sum_phaerr
    '''
    pha_lst = np.array(pha_lst)
    if scal_lst is None:
        scal_lst = np.ones(pha_lst.shape[0])
    scal_lst = np.array(scal_lst)
    assert pha_lst.shape[0] == scal_lst.shape[0], 'PHA number and ratio number do not match!'
    
    # For spectral counts
    pha_scal_lst = pha_lst * scal_lst[:,np.newaxis]
    sum_pha = np.sum(pha_scal_lst, axis=0)
    
    # For spectral counts uncertainties
    # Gaussian error propagation: each channel of `pha_scal_lst` has to have enough photon counts!
    # But this is generally not the case for bkg spectra (scal_lst << 1), so function `add_bkgpha` should be used instead!
    phaerr_lst = np.sqrt(pha_lst) # Poisson statistics
    phaerr_scal_lst = phaerr_lst * scal_lst[:,np.newaxis] # see explanations above
    sum_phaerr = np.sqrt(np.sum(phaerr_scal_lst**2, axis=0)) 
    
    # Write fits file (optional)
    if fits_name is not None:
        hdulist = fits.HDUList()
    
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        channels = np.arange(1,len(sum_pha)+1)
        cols = [fits.Column(name='CHANNEL', format='I', array=channels),
                fits.Column(name='COUNTS', format='J', array=sum_pha),
                fits.Column(name='STAT_ERR', format='D', array=sum_phaerr)]
        hdu_spectrum = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
        hdulist.append(hdu_spectrum)

        # PHA header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, OGIP/92-007: "The OGIP Spectral File Format")
        hdu_spectrum.header['TELESCOP'] = 'STACKED'
        hdu_spectrum.header['INSTRUME'] = 'STACKED'
        hdu_spectrum.header['EXPOSURE'] = expo
        if bkg_file is not None:
            hdu_spectrum.header['BACKFILE'] = bkg_file
        hdu_spectrum.header['BACKSCAL'] = 1.0
        hdu_spectrum.header['CORRSCAL'] = 1.0
        if rmf_file is not None:
            hdu_spectrum.header['RESPFILE'] = rmf_file
        if arf_file is not None:
            hdu_spectrum.header['ANCRFILE'] = arf_file
        hdu_spectrum.header['AREASCAL'] = 1.0
        hdu_spectrum.header['HDUCLASS'] = 'OGIP'
        hdu_spectrum.header['HDUCLAS1'] = 'SPECTRUM'
        hdu_spectrum.header['HDUVERS'] = '1.2.1'
        hdu_spectrum.header['POISSERR'] = False # statistical errors specified in `STAT_ERR` instead
        hdu_spectrum.header['CHANTYPE'] = 'PI'
        hdu_spectrum.header['DETCHANS'] = len(channels)
        hdu_spectrum.header['CREATOR'] = 'XSTACK'
        hdu_spectrum.header['HDUCLAS2'] = 'TOTAL'
        hdu_spectrum.header['HDUCLAS3'] = 'COUNT'
        
        hdulist.writeto(fits_name, overwrite=True)
        
    return sum_pha,sum_phaerr


def add_bkgpha(bkgpha_lst,bkgscal_lst,Ngrp=4,fits_name=None,expo=10):
    '''
    The weighted sum of background PHA files. The weights are specified by `bkgscal_lst`.
    
    Group sources into bins of similar scaling ratio (considering both BACKSCAL and 
    EXPOSURE) For each group, sum the background counts, and compute the uncertainty 
    with Poisson statistics. Then sum the groups, scaling with the averaged scaling 
    ratio, and use Gaussian error propagation.
    
    Parameters
    ----------
    bkgpha_lst: Background PHA file list.
    bkgscal_lst: Scaling ratio list.
    Ngrp: Number of groups with similar scaling ratio. Default is 4.
    
    Returns
    -------
    bkgpha, bkgpha_err
    '''
    bkgpha_lst = np.array(bkgpha_lst)
    bkgscal_lst = np.array(bkgscal_lst)
    assert bkgpha_lst.shape[0] == bkgscal_lst.shape[0], 'number of bkgPHAs and number of scaling ratios do not match!'
    
    # Stacked bkg spectral counts calculated as stacked src spectral counts
    bkgpha, bkgpha_err_UNUSED = add_pha(bkgpha_lst,bkgscal_lst)
    
    # Stacked bkg spectral counts uncertainties estimation: grouping method
    bkggrpflg_lst, bkgscal_ave_lst = make_bkggrpflg(bkgscal_lst,Ngrp=Ngrp) # group bkg spectra with similar scaling ratios
    bkgpha_grp_lst = []
    for i in range(Ngrp):
        bkgpha_tmp = bkgpha_lst[bkggrpflg_lst==i]
        bkgpha_grp_lst.append(add_pha(bkgpha_tmp)[0])
    bkgpha_grp_lst = np.array(bkgpha_grp_lst)
    # then sum the groups (scaling with average scaling ratio, and use Gaussian error propagation)
    bkgpha_UNUSED, bkgpha_err = add_pha(bkgpha_grp_lst,bkgscal_ave_lst)
    
    # write fits file (optional)
    if fits_name is not None:
        hdulist = fits.HDUList()
    
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        channels = np.arange(1,len(bkgpha)+1)
        cols = [fits.Column(name='CHANNEL', format='I', array=channels),
                fits.Column(name='COUNTS', format='D', array=bkgpha), # BKG counts: float
                fits.Column(name='STAT_ERR', format='D', array=bkgpha_err)]
        hdu_spectrum = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
        hdulist.append(hdu_spectrum)

        # PHA header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, OGIP/92-007: "The OGIP Spectral File Format")
        hdu_spectrum.header['TELESCOP'] = 'STACKED'
        hdu_spectrum.header['INSTRUME'] = 'STACKED'
        hdu_spectrum.header['EXPOSURE'] = expo
        hdu_spectrum.header['BACKFILE'] = 'None'
        hdu_spectrum.header['BACKSCAL'] = 1.0
        hdu_spectrum.header['CORRSCAL'] = 1.0
        hdu_spectrum.header['RESPFILE'] = 'None'
        hdu_spectrum.header['ANCRFILE'] = 'None'
        hdu_spectrum.header['AREASCAL'] = 1.0
        hdu_spectrum.header['HDUCLASS'] = 'OGIP'
        hdu_spectrum.header['HDUCLAS1'] = 'SPECTRUM'
        hdu_spectrum.header['HDUVERS'] = '1.2.1'
        hdu_spectrum.header['POISSERR'] = False # statistical errors specified in `STAT_ERR` instead
        hdu_spectrum.header['CHANTYPE'] = 'PI'
        hdu_spectrum.header['DETCHANS'] = len(channels)
        hdu_spectrum.header['CREATOR'] = 'XSTACK'
        hdu_spectrum.header['HDUCLAS2'] = 'BKG'
        hdu_spectrum.header['HDUCLAS3'] = 'COUNT'

        hdulist.writeto('%s'%(fits_name), overwrite=True)
    
    return bkgpha,bkgpha_err


def get_bkgscal(src_file,bkg_file):
    '''
    Get scaling ratio for some background spectrum:
        `scaling ratio = src_areascal / bkg_areascal * src_backscal / bkg_backscal * src_expo / bkg_expo`
        
    Parameters
    ----------
    src_file: source spectrum file name.
    bkg_file: background spectrum file name.
    
    Returns
    -------
    scaling_ratio
    '''
    with fits.open(src_file) as hdu:
        header = hdu['SPECTRUM'].header
        src_expo = header['EXPOSURE']
        src_areascal = header['AREASCAL']
        src_backscal = header['BACKSCAL']
    with fits.open(bkg_file) as hdu:
        bheader = hdu['SPECTRUM'].header
        bkg_expo = bheader['EXPOSURE']
        bkg_areascal = bheader['AREASCAL']
        bkg_backscal = bheader['BACKSCAL']
    bkgscal = src_areascal / bkg_areascal * src_backscal / bkg_backscal * src_expo / bkg_expo
    return bkgscal


def get_expo(src_file):
    with fits.open(src_file) as hdu:
        src_expo = hdu['SPECTRUM'].header['EXPOSURE']
    return src_expo


def make_bkggrpflg(bkgscal_lst,Ngrp=4):
    '''
    Group the `bkgscal_lst` into `Ngrp` groups, according to the scaling ratios. 
    Return an array `bkggrpflg_lst` that tells you which group each background spectrum should be assigned to.
    
    Parameters
    ----------
    bkgscal_lst: The list of scaling-ratio (considering both BACKSCAL and EXPOSURE) for each background spectrum.
    Ngrp: The number of groups to be created. Default is 4.
    
    Returns
    -------
    bkggrpflg_lst: An array that indicates which group each background spectrum should be assigned to. (length = len(bkgscal_lst))
    bkgscal_ave_lst: The average scaling-ratio of each group. (length = `Ngrp`)
    '''
    idx_lst = np.argsort(bkgscal_lst)
    idx_lo = np.array([int(len(idx_lst) / Ngrp * i) for i in range(Ngrp)])
    idx_hi = np.array([int(len(idx_lst) / Ngrp * (i+1)) - 1 for i in range(Ngrp)])
    
    bkggrpflg_lst = np.zeros(len(idx_lst),dtype='int')
    for i in range(len(idx_lst)):
        idx = idx_lst[i]
        mask = (idx <= idx_hi) & (idx >= idx_lo)
        bkggrpflg = np.arange(Ngrp)[mask][0] # the group id of idx
        bkggrpflg_lst[i] = bkggrpflg
    
    bkgscal_ave_lst = np.ones(Ngrp)
    for i in range(Ngrp):
        bkgscal_ave_lst[i] = np.average(bkgscal_lst[bkggrpflg_lst==i])
    
    return bkggrpflg_lst, bkgscal_ave_lst


def first_energy_fits(srcid_lst,first_energy_lst,fits_name):
    '''

    '''
    if fits_name is not None:
        hdulist = fits.HDUList()
        
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        cols = [fits.Column(name='srcid', format='I', array=srcid_lst),
                fits.Column(name='f_energy', format='D', array=first_energy_lst)]
        hdu_fenergy = fits.BinTableHDU.from_columns(cols, name='FENERGY')
        hdulist.append(hdu_fenergy)

        hdulist.writeto('%s'%(fits_name), overwrite=True)
    return 
