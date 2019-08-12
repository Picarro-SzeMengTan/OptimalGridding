This repository contains the code and tutorials associated with the paper "Optimal gridding and degridding in radio interferometry imaging" by Haoyang Ye, Stephen F. Gull, Sze M. Tan and Bojan Nikolic, submitted to the Monthly Notices of the Royal Astronomical Society. A preliminary version is available at https://arxiv.org/abs/1906.07102.

Within the notebooks directory are a number of Jupyter notebooks which are intended as tutorials describing the use and properties of the least misfit functions. These are convolutional gridding functions designed to minimize the difference between a map computed using direct Fourier transform and one computed using a fast Fourier trasform. The code for generating and analyzing these functions is in the file notebooks/algorithms/core.py.

The notebooks are intended to be read in the following order:

- Quick Start
- Evaluating Performance of Gridding and Grid Correction Functions
- Optimization to Find Least Misfit Functions
- Table Lookup and Interpolation of Gridding Convolution Function

The notebooks and code are intended for use with Python 3.5 and above and make use of the numpy, scipy and attrs libraries.


Sze Tan
(szemengtan@gmail.com)