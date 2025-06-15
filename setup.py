from setuptools import setup, find_packages
import os

with open("VERSION") as f:
    lines = f.readlines()
    version = lines[0].strip()
    lastupdate = lines[1].strip()

os.system("cp VERSION ./Xstack/")
os.system("cp VERSION ./Xstack_scripts/")

setup(
    name="Xstack",
    version=version,
    description="An X-ray Spectral Shifting and Stacking Code",
    author="Shi-Jiang Chen, Johannes Buchner and Teng Liu",
    author_email="JohnnyCsj666@gmail.com",
    url="https://github.com/AstroChensj/Xstack.git",
    #packages=find_packages(),
    packages=["Xstack","Xstack_scripts"],
    install_requires=[
        "astropy",
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "numba",
        "sfdmap",
        "joblib",
    ],
    package_data={"": ["**/tbabs_1e20.txt"],
                  "Xstack": ["fkspec_sh/*.sh"]},
    entry_points={
        "console_scripts": [
            "runXstack = Xstack_scripts.Xstack_autoscript:main"
        ]
    }
)
