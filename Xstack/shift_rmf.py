import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from tqdm import tqdm
import os


def shift_rmf(mat,ebo,z,rmfsft_method='PAR'):
    '''
    Shift the RMF. This is literally done by two steps: 
    1) Shift in the direction of output channel energy. That is to say, shift and broaden the probability profile for 
       each input energy (i.e. when the detector receive a photon with some input energy, the probability that a signal 
       at some output channel energy will be observed; so this is a function of output channel energy) by (1+z); 
    2) Shift in the direction of input energy. That is to say, shift the input energy by (1+z).
    
    Parameters
    ----------
    mat : FITS_rec
        The `MATRIX` HDU data from a standard RMF file.
    ebo : FITS_rec
        The `EBOUNDS` HDU data from a standard RMF file.
    z : float
        Redshift.
    rmfsft_method : str
        The RMF shifting method. Two methods are available:
        * `PAR`: Parameterized method, i.e. approximate the probability profile with a Gaussian, and shift the Gaussians.
        * `NONPAR`: Non-PARameterized method, i.e. shift the probability profile directly. This should be more accurate, 
          and takes into account the off-diagonal elements in the RMF matrix. However, the non-PARameterized method is 
          far more time-consuming than PARameterized method (~10^2 times slower).
          
    Returns
    -------
    prob_sft : ndarray
        The shifted probability matrix.
    '''
    iene_lo = mat['ENERG_LO']
    iene_hi = mat['ENERG_HI']
    iene_ce = (iene_lo + iene_hi) / 2
    iene_wd = (iene_hi - iene_lo)
    iene_id = np.arange(len(iene_ce))
    
    ene_lo = ebo['E_MIN']
    ene_hi = ebo['E_MAX']
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = (ene_hi - ene_lo)
    ene_id = np.arange(len(ene_ce))
    
    if rmfsft_method == 'PAR': # Parameterized method
        # get dspmap
        dspmap = 'ene_dsp.fits'
        if not os.path.exists(dspmap):
            print('dspmap `ene_dsp.fits` not found: will automatically generate one with `MATRIX` and `EBOUNDS` you provide...')
            make_dspmap(mat, ebo, dspmap)
        
        with fits.open(dspmap) as hdu:
            dsp = hdu[1].data
        norm = dsp['norm']
        ene_nom = dsp['ene_nom']
        ene_dsp = dsp['ene_dsp']
        
        # get prob_lst
        grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
        prob_sft = np.zeros(grid[0].shape)
        
        iene_ubound = iene_lo.max()
        iene_lbound = iene_hi.min()
        for i in range(len(iene_ce)):
            iene_lo_map = iene_lo[i] * (1+z)
            iene_hi_map = iene_hi[i] * (1+z)
            
            if iene_lo_map > iene_ubound:
                continue
            if iene_hi_map < iene_lbound:
                continue
                
            # step 1: output energy *(1+z), dispersion *(1+z)
            prob_sft_i = gaussian(ene_ce, norm[i]/(1+z), ene_nom[i]*(1+z), ene_dsp[i]*(1+z)) * ene_wd
            
            # step 2: input energy *(1+z)
            mask = (iene_lo_map < iene_hi) & (iene_hi_map > iene_lo)
            iene_id_mask = iene_id[mask]
            iene_wd_mask = iene_wd[mask]
            iene_lo_mask = iene_lo[mask]
            iene_hi_mask = iene_hi[mask]
            
            iene_wd_mask[0] = iene_hi_mask[0] - iene_lo_map
            iene_wd_mask[-1] = iene_hi_map - iene_lo_mask[-1]
            
            prob_mask = iene_wd_mask / iene_wd_mask.sum()
            
            for j in range(len(iene_id_mask)):
                prob_sft[iene_id_mask[j]] += prob_sft_i * prob_mask[j]
                
    elif rmfsft_method == 'NONPAR': # Non-PARameterized method
        # get prob_lst
        grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
        prob = np.zeros(grid[0].shape) # probability per channel
        
        n_grp = mat['N_GRP']
        f_chan = mat['F_CHAN']
        n_chan = mat['N_CHAN']
        mat = mat['MATRIX']
        
        f_chan_0 = int(np.min([np.min(f_chan[_]) for _ in range(len(f_chan))])) # the zero point of channel index
        for i in range(len(iene_ce)):
            f_mat = 0
            for grp_j in range(n_grp[i]):
                f_chan_j = f_chan[i][grp_j] - f_chan_0
                n_chan_j = n_chan[i][grp_j]
                e_chan_j = f_chan_j + n_chan_j # ending index of group_j in channel
                e_mat = f_mat + n_chan_j # ending index of group_j in matrix[i]
                
                prob[i][f_chan_j:e_chan_j] += mat[i][f_mat:e_mat]
                f_mat += n_chan_j
        
        prob_sft = np.zeros(grid[0].shape)
        iene_ubound = iene_lo.max()
        iene_lbound = iene_hi.min()
        for i in range(len(iene_ce)):
            # input energy *(1+z)
            iene_lo_map = iene_lo[i] * (1+z)
            iene_hi_map = iene_hi[i] * (1+z)
            
            if iene_lo_map > iene_ubound:
                continue
            if iene_hi_map < iene_lbound:
                continue
                
            # step 1: output energy *(1+z), dispersion *(1+z)
            prob_1d = np.zeros(len(ene_ce))
            ene_ubound = ene_lo.max()
            ene_lbound = ene_hi.min()
            for j in range(len(ene_ce)):
                ene_lo_map = ene_lo[j] * (1+z)
                ene_hi_map = ene_hi[j] * (1+z)
                
                if ene_lo_map > ene_ubound:
                    continue
                if ene_hi_map < ene_lbound:
                    continue
                
                mask = (ene_lo_map < ene_hi) & (ene_hi_map > ene_lo)
                ene_id_mask = ene_id[mask]
                ene_wd_mask = ene_wd[mask]
                ene_lo_mask = ene_lo[mask]
                ene_hi_mask = ene_hi[mask]
                
                ene_wd_mask[0] = ene_hi_mask[0] - ene_lo_map
                ene_wd_mask[-1] = ene_hi_map - ene_lo_mask[-1]
                
                prob_mask = ene_wd_mask / ene_wd_mask.sum()
                
                for k in range(len(ene_id_mask)):
                    prob_1d[ene_id_mask[k]] += prob[i][j] * prob_mask[k]
                prob_1d[ene_id_mask[0]:ene_id_mask[-1]+1] += prob[i][j] * prob_mask
                
            # step 2: input energy * (1+z)
            mask = (iene_lo_map < iene_hi) & (iene_hi_map > iene_lo)
            iene_id_mask = iene_id[mask]
            iene_wd_mask = iene_wd[mask]
            iene_lo_mask = iene_lo[mask]
            iene_hi_mask = iene_hi[mask]
            
            iene_wd_mask[0] = iene_hi_mask[0] - iene_lo_map
            iene_wd_mask[-1] = iene_hi_map - iene_lo_mask[-1]
            
            prob_mask = iene_wd_mask / iene_wd_mask.sum()
            
            for l in range(len(iene_id_mask)):
                prob_sft[iene_id_mask[l]] += prob_1d * prob_mask[l]
        
    else:
        raise Exception('Available rmfsft_method (RMF shifting method): `PAR` or `NONPAR` (see help(shift_rmf) for illustration)!')
            
    return prob_sft


def add_rmf(prob_lst,arf_file,expo_lst=None,fits_name=None,sample_rmf='sample.rmf',srcid_lst=None):
    '''
    Sum the shifted RMFs together. Each RMF is assigned the same weighting factor as ARF. 
    
    Parameters
    ----------
    prob_lst : list or array_like
        The probability matrix list.
    arf_file : str
        The shifted&stacked ARF file name. Should be the product of function `add_arf`! 
    expo_lst : list or array_like, optional
        Exposure list.
    fits_name : str, optional
        Output RMF name.
    sample_rmf : str, optional
        Sample RMF name.
    srcid_lst : list or array_like, optional
        Source ID list.
    
    Returns
    -------
    sum_prob : ndarray
        The stacked RMF matrix.
    '''
    if expo_lst is None:
        expo_lst = np.ones(len(prob_lst))
    if srcid_lst is None:
        srcid_lst = np.arange(len(prob_lst))
    expo_lst = np.array(expo_lst)
    srcid_lst = np.array(srcid_lst)
    
    # rmf is to be weighted in the same way as arf
    with fits.open(arf_file) as hdu:
        scal = hdu['ARFSCAL'].data
    arfscal_lst = scal['ARFSCAL']
    rmfscal_lst = arfscal_lst
    
    # add rmf
    prob_lst = np.array(prob_lst)
    prob_lst = prob_lst * rmfscal_lst[:,np.newaxis,np.newaxis]
    sum_prob = np.sum(prob_lst,axis=0)
    
    # give special treat to: extremely small probabilities, empty probability, NaN, and normalization issues
    sum_prob /= np.sum(sum_prob,axis=1)[:,np.newaxis] # normalize
    sum_prob[np.isclose(sum_prob,0,rtol=1e-06, atol=1e-06, equal_nan=False)] = 0 # remove elements with probability below the 1e-6 threshold
    sum_prob[np.isnan(sum_prob)] = 0 # remove NaN
    sum_prob[sum_prob<0] = 0 # remove negative elements
    sum_prob /= np.sum(sum_prob,axis=1)[:,np.newaxis] # renormalize
    sum_prob[np.isnan(sum_prob)] = 0 # remove NaN (produced when 0/0)
    # for the first few input energies, the probability may be empty
    # assign the first channel with 1 (an arbitrary choice)
    for i in range(len(sum_prob)):
        if np.max(sum_prob[i]) == 0.:
            sum_prob[i][0] = 1
    
    # write fits file
    if fits_name is not None:
        hdulist = fits.HDUList()
        
        # extension 0: primary hdu
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        # extension 1: MATRIX
        with fits.open(sample_rmf) as hdu:
            rmf = hdu[1].data
            ebo = hdu[2].data
        iene_lo = rmf['ENERG_LO']
        iene_hi = rmf['ENERG_HI']
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
        hdu_matrix.header['DETCHANS'] = prob_lst[0].shape[1]
        hdu_matrix.header['HDUCLASS'] = 'OGIP'
        hdu_matrix.header['HDUCLAS1'] = 'RESPONSE'
        hdu_matrix.header['HDUCLAS2'] = 'RSP_MATRIX'
        hdu_matrix.header['HDUVERS'] = '1.3.0'
        hdu_matrix.header['TLMIN4'] = 1 # the first channel in the response
        hdu_matrix.header['EXPOSURE'] = expo_lst.sum()
        hdu_matrix.header['ANCRFILE'] = arf_file
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
        hdu_ebounds.header['DETCHANS'] = prob_lst[0].shape[1]
        hdu_ebounds.header['HDUCLASS'] = 'OGIP'
        hdu_ebounds.header['HDUCLAS1'] = 'RESPONSE'
        hdu_ebounds.header['HDUCLAS2'] = 'EBOUNDS'
        hdu_ebounds.header['HDUVERS'] = '1.2.0'
        hdulist.append(hdu_ebounds)
        
        # extension 3: RMFSCAL
        cols = [fits.Column(name='SRCID', format='J', array=srcid_lst),
                fits.Column(name='RMFSCAL', format='D', array=rmfscal_lst)]
        hdu_rmfscal = fits.BinTableHDU.from_columns(cols, name='RMFSCAL')
        hdulist.append(hdu_rmfscal)
        
        hdulist.writeto('%s'%(fits_name), overwrite=True)
        
    return sum_prob


#==============================================
############ Make Dispersion Map ##############
#==============================================
def gaussian(x, amplitude, mean, stddev):
    '''
    A gaussian function.

    Parameters
    ----------
    x : float or array_like
    amplitude : float
    mean : float
    stddev : float

    Returns
    -------
    pdf : float or ndarray
    '''
    pdf = amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)
    return pdf


def get_ene_dsp(ene_ce,prob_lst,fixed_mean=True):
    '''
    Get energy dispersion.

    Parameters
    ----------
    ene_ce : array_like
        Output channel energy.
    prob_lst : array_like
        Probability profile for some input energy (this is a function of output channel energy). Must have same length as `ene_ce`.
    fixed_mean : bool
        If true, the mean energy of the Gaussian will be fixed at the nominal energy (which corresponds to maximal probability).

    Returns
    -------
    norm : float
        The Gaussian normalization.
    ene_nom : float
        The Gaussian central energy (this is the nominal energy for some input energy).
    ene_dsp : float
        The Gaussian width (this is the energy dispersion for some input energy).
    '''
    ene_nom = ene_ce[np.argmax(prob_lst)] # nominal energy
    if fixed_mean:
        mlo = ene_nom # lower bound for mean
        mhi = ene_nom + 1e-6 # upper bound for mean
    else:
        mlo = 0
        mhi = np.inf
    popt, pcov = curve_fit(gaussian, ene_ce, prob_lst,
                           p0=[1, ene_nom, 1], bounds=([0, mlo, 0], [np.inf, mhi, np.inf]))
    norm = popt[0]
    ene_dsp = popt[2]
    return norm,ene_nom,ene_dsp


def make_dspmap(mat,ebo,out_name):
    '''
    Make energy dispersion map.
    
    Parameters
    ----------
    mat : FITS_rec
        The `MATRIX` HDU data from a standard RMF file.
    ebo : FITS_rec
        The `EBOUNDS` HDU data from a standard RMF file.
    out_name : str
        The output dispersion map name.

    Returns
    -------
    None.
    '''
    iene_lo = mat['ENERG_LO']
    iene_hi = mat['ENERG_HI']
    iene_ce = (iene_lo + iene_hi) / 2
    iene_wd = (iene_hi - iene_lo)
    
    ene_lo = ebo['E_MIN']
    ene_hi = ebo['E_MAX']
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = (ene_hi - ene_lo)
    
    # get prob_lst
    grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
    prob = np.zeros(grid[0].shape) # probability per channel
    
    n_grp = mat['N_GRP']
    f_chan = mat['F_CHAN']
    n_chan = mat['N_CHAN']
    mat = mat['MATRIX']
    
    f_chan_0 = int(np.min([np.min(f_chan[_]) for _ in range(len(f_chan))])) # the zero point of channel index
    for i in range(len(iene_ce)):
        f_mat = 0
        for grp_j in range(n_grp[i]):
            f_chan_j = f_chan[i][grp_j] - f_chan_0
            n_chan_j = n_chan[i][grp_j]
            e_chan_j = f_chan_j + n_chan_j # ending index of group_j in channel
            e_mat = f_mat + n_chan_j # ending index of group_j in matrix[i]
            
            prob[i][f_chan_j:e_chan_j] += mat[i][f_mat:e_mat]
            f_mat += n_chan_j
    
    prob_ene = prob / ene_wd # probability per energy bin
    
    # get nominal energy and energy dispersion
    print('################# Generating dspmap ###################')
    norm = []
    ene_nom = []
    ene_dsp = []
    for i in tqdm(range(len(iene_ce))):
        norm_i,ene_nom_i,ene_dsp_i = get_ene_dsp(ene_ce,prob_ene[i],fixed_mean=True)
        norm_i /= (gaussian(ene_ce,norm_i,ene_nom_i,ene_dsp_i)*ene_wd).sum() # renormalize the gaussian profile
        norm.append(norm_i)
        ene_nom.append(ene_nom_i)
        ene_dsp.append(ene_dsp_i)
    norm = np.array(norm)
    ene_nom = np.array(ene_nom)
    ene_dsp = np.array(ene_dsp)
    
    # make fits file
    column_names = ['ENERG_LO','ENERG_HI','norm','ene_nom','ene_dsp']
    formats = ['D','D','D','D','D']
    arrays = [iene_lo,iene_hi,norm,ene_nom,ene_dsp]
    columns = [fits.Column(name=col_name,format=format_,array=array_) for col_name,format_,array_ in zip(column_names,formats,arrays)]
    coldefs = fits.ColDefs(columns)
    table = fits.BinTableHDU.from_columns(coldefs)
    table.writeto(out_name,overwrite=True)
    print('########### dspmap successfully generated! ############')
    
    return
