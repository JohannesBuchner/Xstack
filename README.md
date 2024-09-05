# Xstack
## What is <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> ?

<span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is a open-source, light-weight code for **X-ray spectral shifting and stacking**. 

Given the ***redshift*** of each spectrum in a list, <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> will first shift these spectra **(PHA, counts vs. output energy (channel))** from observed-frame to rest-frame, and then sum them together. The response matrix files *(RMF, the probability that a photon with input energy $E_1$ will be detected with an output energy $E_2$)* and ancillary response files *(ARF, effective area vs. input energy)* are shifted and stacked in a similar way to the spectrum. 

<span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> also supports correction of Galactic absorption, if an additional ***NH*** value (in units of 1 $\text{cm}^{-2}$) for each spectrum is given.



## Prerequisites and Installation

To install <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> (along with the prerequisite packages), simply put:
```shell
git clone https://github.com/AstroChensj/Xstack.git
cd Xstack
pip install .
```
In order to activate the `bootstrap` feature, you will also need `GNU parallel`, a fantastic tool for parallel computation. See [their website](https://savannah.gnu.org/news/?id=10666) for the installation of `parallel` (*within 5s*).

## How to use <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>
Currently there are 2 ways to use  <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>:

+ Call the  <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> module inside *Python*;
	* View [`./demo/demo.ipynb`](https://nbviewer.org/github/AstroChensj/Xstack/blob/main/demo/demo.ipynb) for a quick walk-through!
+ Or run the *Python* script `./scripts/runXstack.py`.
	* A simple example would be:
		```shell
		python ./scripts/runXstack.py filelist.txt 1.0 2.3 SHP --outsrc=stack.pi --outbkg=stackbkg.pi --outrmf=stack.rmf --outarf=stack.arf --parametric_rmf=True --num_bkg_groups=10
		```
		~~The illustration to be filled in later ...~~

### Bootstrap (experimental)

<span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> also supports bootstrap feature, in order to examine if some few bright sources could potentially dominate the stacked results. This is basically done by bootstrapping the original sample for `num_bootstrap` times (specified by the user), and produce `num_bootstrap` stacked spectra (spectra, ARFs, RMFs).

For now, the bootstrap feature is only achievable through the second way, i.e. running the python script `runXstack.py`. Here is how:
```shell
python ./scripts/runXstack.py filelist.txt 1.0 2.3 SHP --num_bkg_groups=10 --num_bootstrap=100 --pre_bootstrap=bs
```
~~The illustration to be filled in later ...~~

Note that after this command, <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> only produces a shell script (`bs.sh`), to be run by the user with `parallel` command:

```shell
parallel -j 20 -a bs.sh
```
Here `20` is the number of cores to be used in parallel, and you can modify that. 