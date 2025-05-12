import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from Xstack.misc import make_grpflg,rebin_pi


fig, ax1 = plt.subplots(1,1,figsize=(4,4),dpi=200)
color_left = 'red'
color_right = 'blue'

fene_name = 'fene_c038_large.fits'
with fits.open(fene_name) as hdu:
    data = hdu['FENERGY'].data
arffene = data['arffene']
fene = data['fene']

sorted_arffene = np.sort(arffene)
cdf_arffene = np.arange(1,len(sorted_arffene)+1)/len(sorted_arffene)
extendene = np.logspace(-1,1,1000)
extendcdf_arffene = np.interp(extendene,sorted_arffene,cdf_arffene,left=0,right=1)

sorted_fene = np.sort(fene)
cdf_fene = np.arange(1,len(sorted_fene)+1)/len(sorted_fene)
extendene = np.logspace(-1,1,1000)
extendcdf_fene = np.interp(extendene,sorted_fene,cdf_fene,left=0,right=1)


src_file = 'src_c038_large.fits'
grp_file = 'grp_c038_large.fits'
bkg_file = 'bkg_c038_large.fits'
rmf_file = 'rmf_c038_large.fits'
with fits.open(src_file) as hdu:
    data = hdu['SPECTRUM'].data
chan = data['CHANNEL']
pha = data['COUNTS']
phaerr = np.sqrt(pha)
with fits.open(bkg_file) as hdu:
    data = hdu['SPECTRUM'].data
chan = data['CHANNEL']
bkgpha = data['COUNTS']
bkgphaerr = np.sqrt(pha)
with fits.open(rmf_file) as hdu:
    mat = hdu['MATRIX'].data
    ebo = hdu['EBOUNDS'].data
ene_lo = ebo['E_MIN']
ene_hi = ebo['E_MAX']
ene_ce = (ene_lo + ene_hi) / 2
ene_wd = ene_hi - ene_lo

eene = np.logspace(np.log10(0.2),np.log10(ene_ce.max()),18)
eelo = eene[:-1]
eehi = eene[1:]
make_grpflg(src_file,grp_file,method='EDGE',rmf_file=rmf_file,eelo=eelo,eehi=eehi)
with fits.open(grp_file) as hdu:
    data = hdu[1].data
grpflg = data['GROUPING']
grpene_lo,grpene_hi,grppha,grpphaerr = rebin_pi(ene_lo,ene_hi,pha,phaerr,grpflg)
grpene_lo,grpene_hi,grpbkgpha,grpbkgphaerr = rebin_pi(ene_lo,ene_hi,bkgpha,bkgphaerr,grpflg)
grpbkgfrac = grpbkgpha/grppha
grpene_wd = grpene_hi - grpene_lo
grpene_ce = (grpene_lo + grpene_hi) / 2


ax1.plot(extendene,extendcdf_arffene,ls='-',c=color_left,label='ARF')
ax1.plot(extendene,extendcdf_fene,ls='--',c=color_left,label='PHA')
ax1.legend(fontsize=10)
ax1.fill_betweenx(y=[0,1.1], x1=0.1, x2=0.4, color='gray', alpha=0.2, zorder=10)
ax1.fill_betweenx(y=[0,1.1], x1=8.0, x2=10.0, color='gray', alpha=0.2, zorder=10)

ax1.set_xscale('log')
ax1.set_xlim(0.2,10)
x_ticks = [0.2, 0.4, 1.0, 2.3, 4.0, 8.0]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([str(x) for x in x_ticks])
ax1.set_xlabel('Energy (keV)',fontsize=10)
ax1.tick_params("x",which="major",
                length=10,width = 1.0,size=5,labelsize=8,pad=3)
ax1.tick_params("x",which="minor",
                length=10,width = 1.0,size=5,labelsize=8,pad=3)
ax1.set_ylim(0,1.1)
ax1.set_ylabel('Frac sources',fontsize=10,color=color_left)
ax1.tick_params("y",which="major",
                length=10,width = 1.0,size=5,labelsize=8,pad=3,labelcolor=color_left)
ax1.tick_params("y",which="minor",
                length=10,width = 1.0,size=5,labelsize=8,pad=3,labelcolor=color_left)


ax2 = ax1.twinx()
ax2.plot(grpene_ce,1-grpbkgfrac,c=color_right)
ax2.set_ylim(0,1.1)
ax2.set_ylabel('Net/Total counts',fontsize=10,color=color_right)
ax2.tick_params("y",which="major",
                length=10,width = 1.0,size=5,labelsize=8,pad=3,labelcolor=color_right)
ax2.tick_params("y",which="minor",
                length=10,width = 1.0,size=5,labelsize=8,pad=3,labelcolor=color_right)