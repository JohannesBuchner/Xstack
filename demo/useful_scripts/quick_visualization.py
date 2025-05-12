import numpy as np
from matplotlib import pyplot as plt
from Xstack.misc import make_grpflg,make_dataarf_plot

src_name = "stacked.pi"
grp_name = "stacked_grp.pi"
bkg_name = "stacked_bkg.pi"
arf_name = "stacked.arf"
rmf_name = "stacked.rmf"

# make group flag
eene = np.logspace(np.log10(0.2),np.log10(12),18)
eelo = eene[:-1]
eehi = eene[1:]
make_grpflg(src_name,grp_name,method='EDGE',rmf_file=rmf_name,eelo=eelo,eehi=eehi)

# make data/arf plot
fig, ax1 = plt.subplots(figsize=(8, 6))
make_dataarf_plot(src_name,bkg_name,arf_name,rmf_name,grp_name,normalize_at=4,plot=True,ax=ax1) # this plots EF(E), or equivalently leed in xspec; a powerlaw with photon index of 2 should look flat in this plot