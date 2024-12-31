from setuptools import setup, find_packages

setup(
    name='Xstack',
    version='0.1.0',
    description='An X-ray Spectral Shifting and Stacking Code',
    author='Shi-Jiang Chen, Johannes Buchner and Teng Liu',
    author_email='JohnnyCsj666@gmail.com',
    url='https://github.com/AstroChensj/Xstack.git',
    #packages=find_packages(),
    packages=['Xstack','scripts'],
    install_requires=[
        'astropy',
        'numpy',
        'scipy',
        'pandas',
        'tqdm',
        'numba',
        'sfdmap',
        'joblib',
    ],
    package_data={'': ['**/tbabs_1e20.txt'],
                  'Xstack': ["fkspec_sh/*.sh"]},
    #scripts=['scripts/runXstack.py'],
    entry_points={
        'console_scripts': [
            'runXstack = scripts.Xstack_autoscript:main'
        ]
    }
)
