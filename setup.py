# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 02.03.2021, setup.py
#

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='trockenmauer',
    version='0.0.1',
    author='Stefan Hochuli',
    description='Mth Trockenmauer',
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy', 'matplotlib==3.4.2', 'rtree'
    ],
    python_requires='>=3.8',
    url='https://github.com/hochstibe/mth',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
