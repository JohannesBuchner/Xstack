from setuptools import setup, find_packages

setup(
    name='Xstack',
    #version='0.1.0',
    description='An X-ray Spectral Shifting \& Stacking Code',
    author='Shi-Jiang Chen, Johannes Buchner \& Teng Liu',
    author_email='JohnnyCsj666@gmail.com',
    #url='https://github.com/yourusername/Xstack',
    packages=find_packages(),
    install_requires=[
        'astropy',
        'numpy',
        'scipy',
        'tqdm',
    ],
)
