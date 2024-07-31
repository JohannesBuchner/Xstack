from setuptools import setup, find_packages

setup(
    name='Xstack',
    version='0.1.0',
    description='An X-ray Spectral Shifting and Stacking Code',
    author='Shi-Jiang Chen, Johannes Buchner and Teng Liu',
    author_email='JohnnyCsj666@gmail.com',
    url='https://github.com/AstroChensj/Xstack.git',
    #packages=find_packages(),
    packages=['Xstack'],
    install_requires=[
        'astropy',
        'numpy',
        'scipy',
        'tqdm',
        'sfdmap',
    ],
)
