import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.integrate import dblquad


class Distribution(object):
    """
    from: https://stackoverflow.com/questions/21100716/fast-arbitrary-distribution-random-sampling-inverse-transform-sampling    
    
    NOTE: the functions sample_unnormalized_pdf_1d and sample_unnormalized_pdf_2d below
          facilitate the usage of this class. Use them preferably

    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]
    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)
        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)
        return self.transform(index)

def sample_unnormalized_pdf_1d(updf, xmin, xmax, n_samples):
    """
    samples from an unnormalized pdf function
    updf: the function (must return values >0), accepting one argument (vector)
    xmin, xmax: the range within which samples are to be generated
                pdf will be normalized to 1 within this range
                probability outside this range is assumed to be zero
    n_samples: the number of samples to generate

    ----
    example usage
    ----
        >> from scipy.integrate import dblquad, quad
        >> xmin, xmax = 0,800
        >> def f(x):
        >>     return (x>=0)*np.exp(-5e-3*x) + np.exp(-(x-150)**2/2000)
        >> Z = quad(f, xmin, xmax)[0] # normalization constant, only for plotting
        >> k = sample_unnormalized_pdf_1d(f, xmin, xmax, n_samples=100000)
        >> xr = np.linspace(xmin,xmax,1000)
        >> plt.hist(k,density=True,bins=100, alpha=.5) 
        >> plt.plot(xr, f(xr)/Z, color="black")
    """
    n = 1000 # the resolution of the sampling
    xr = np.linspace(xmin, xmax, n)
    return Distribution(updf(xr), transform=lambda x:x*(xmax-xmin)/n+xmin)(n_samples)[0]

def sample_unnormalized_pdf_2d(updf, xmin, xmax, ymin, ymax, n_samples):
    """
    samples from an unnormalized pdf function
    updf: the function (must return values >0), accepting two arguments (vectors)
    xmin, xmax, ymin, ymax: 
                the range within which samples are to be generated
                assumes pdf will be normalized to 1 within this range
                probability outside this range is assumed to be zero
    n_samples: the number of samples to generate

    ----
    example usage
    ----
        >> from scipy.integrate import dblquad, quad
        >> xmin, xmax, ymin, ymax = 2,10,-10,-2
        >> def f(x,y):
        >>     return np.abs(x+y**3/100)
        >> k = sample_unnormalized_pdf_2d(f, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, n_samples=10000)
        >> plt.hist2d(k[:,1], k[:,0], bins=100);
        >> plt.colorbar()
    """
    n = 100 # the sqrt of the resolution of the sampling
    v0 = np.linspace(xmin, xmax, n)
    v1 = np.linspace(ymin, ymax, n)
    V0,V1 = np.meshgrid(v0, v1)

    rnge = np.r_[[[ymax-ymin, xmax-xmin]]].T
    rmin = np.r_[[[ymin, xmin]]].T
    return Distribution(updf(V0,V1), transform=lambda x: x*rnge/n+rmin)(n_samples).T
