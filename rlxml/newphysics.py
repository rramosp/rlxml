import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from progressbar import progressbar as pbar
from rlxutils.optimization import coordinate_descent_minimize
import tensorflow as tf
from joblib import Parallel, delayed
import sys
import rlxutils
from joblib import Parallel, delayed

"""
code for reproducing procedures in

    Asymptotic formulae for likelihood-based tests of new physics
    https://arxiv.org/pdf/1007.1727.pdf

"""

def distplot(x, pdf=None, pdf_name=None, pdf_color="black", **kwargs):
    counts = plt.hist(x, density=True, **kwargs)[0]
    xmin, xmax = np.min(x), np.max(x)
    
    if pdf is not None:
        xr = np.linspace(xmin, xmax, 100)
        plt.plot(xr, pdf(xr), color=pdf_color, label=pdf_name)
        
    plt.ylim(0,np.max(counts)*1.05)
    plt.grid();
    if pdf_name is not None:
        plt.legend();

    return xmin, xmax

def kdensity_smoothed_histogram(x):
    x = pd.Series(x).dropna().values
    t = np.linspace(np.min(x), np.max(x), 100)
    p = kdensity(x)(t)
    return t, p

def kdensity(x):
    import numbers
    if len(x.shape) != 1:
        raise ValueError("x must be a vector. found "+str(x.shape)+" dimensions")
    stdx = np.std(x)
    bw = 1.06*stdx*len(x)**-.2 if stdx != 0 else 1.
    kd = KernelDensity(bandwidth=bw)
    kd.fit(x.reshape(-1, 1))

    func = lambda z: np.exp(kd.score_samples(np.array(z).reshape(-1, 1)))
    return func


def plot_kdensity_smoothed_histogram(x, plot_equivalent_gaussian=False, plot_equivalent_poisson=False):
    plt.plot(*kdensity_smoothed_histogram(x), color="black", lw=2, label="data")
    
    if plot_equivalent_gaussian:
        m,s = np.mean(x), np.std(x)
        p = np.percentile(x, [1,99])
        xr = np.linspace(p[0], p[1], 100)
        plt.plot(xr, stats.norm(loc=m, scale=s).pdf(xr), color="blue", alpha=.5, lw=2, label="equiv gaussian")  
    if plot_equivalent_poisson:
        assert x.dtype==int, "for plotting poisson equivalent your data must be composed of integers"
        
        m,v = np.mean(x), np.std(x)
        ep = stats.poisson(loc=np.round(m-v**2,0), mu=v**2)
        ks = pd.Series(x).value_counts().sort_index()/len(x)
        plt.plot(ks.index, ep.pmf(ks.index.values), color="red", label="equiv poisson")
                

from scipy import special
gaus_pdf = lambda x, mu, sigma: np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))    
gaus_cdf = lambda x, mu, sigma: 0.5*(1+special.erf((x-mu)/(sigma*sqrt2)))

exp_pdf  = lambda x, t: t * np.exp(-t*x)   
exp_cdf  = lambda x, t: 1 - np.exp(-t*x)       

log_factorial = np.vectorize(lambda k: np.sum(np.log(range(1,k+1))))

def get_sgbg_probs(mu):
    """
    scales p_s by mu and then normalizes with the corresponding complementary p_b
    """
    assert mu>=0, "mu must be >=0"
    p_s, p_b = mu, 1 
    z = p_s + p_b
    p_s, p_b = p_s/z, p_b/z
    return p_s, p_b

class SignalBg_BinnedModel:
    def __init__(self, t, mu_s, sigma_s, mu, bin_edges, n_events):

        self.t, self.mu_s, self.sigma_s, self.mu = t, mu_s, sigma_s, mu        
        self.bin_edges = bin_edges
        self.n_bins = len(bin_edges)-1
        self.n_events = n_events
        self.init_distributions()

    def get_params(self, include_mu=False):
        r = {'t': self.t, 'mu_s': self.mu_s, 'mu_sigma': self.sigma_s, 'bin_edges': self.bin_edges, 'n_events': self.n_events }
        if include_mu:
            r['mu'] = self.mu
        return r

    def init_distributions(self):
        self.b = stats.expon(scale=1/self.t)
        self.s = stats.norm(loc=self.mu_s, scale=self.sigma_s)     

        self.p_s, self.p_b = get_sgbg_probs(self.mu)
        self.s_tot = int(self.n_events * self.p_s)
        self.b_tot = int(self.n_events - self.s_tot)
        
        #self.si = self.s_tot*pd.Series([self.s.cdf(i) for i in self.bin_edges]).diff().dropna().values
        #self.bi = self.b_tot*pd.Series([self.b.cdf(i) for i in self.bin_edges]).diff().dropna().values

        self.compute_sibi()

        # compute distribution for each bin
        self.bins_distributions = [stats.poisson(mu=self.si[i]+self.bi[i]) for i in range(len(self.bi))]       

    def compute_sibi(self):
        self.si = self.s_tot*pd.Series([self.s.cdf(i) for i in self.bin_edges]).diff().dropna().values
        self.bi = self.b_tot*pd.Series([self.b.cdf(i) for i in self.bin_edges]).diff().dropna().values


    def clone(self, new_mu=None):
        return self.__class__(self.t, self.mu_s, self.sigma_s, self.mu if new_mu is None else new_mu, self.bin_edges, self.n_events)

    def set_mu(self, mu):
        self.mu = mu
        self.init_distributions()
        return self

    def rvs(self):
        """
        sample from the distribution of each bin
        """
        return np.r_[[i.rvs(1).sum() for i in self.bins_distributions]]        

    def log_prob(self, x):
        """
        log prob = log prod prob(bin_i) = sum log prob(bin_i) 
        $$\log \prod_{j=1}^N \frac{(\mu s_j + b_j)^{n_j}}{n_j!}e^{-\mu s_j + b_j} = \
        \sum_{j=1}^N n_j \log (\mu s_j + b_j) - \log(n_j!) - \mu s_j + b_j $$
        """
        term = self.si+self.bi
        return np.sum(x*np.log(term) - log_factorial(x) - term)

    def get_mu_MLE(self, x):
        """
        MLE estimator for mu and observed histogram x, with the rest of the params fixed
        """
        f = lambda mu: -self.clone().set_mu(mu).log_prob(x) if mu>=0 else 1000-100*mu
        
        r = minimize(f, np.random.random(), method="Nelder-Mead")
        assert r.success, r.message
        
        return r.x[0]

    def get_t_mu(self, x):
        """
        t_mu = -2 * lkhood(mu)/lkhood(mu_hat)
        mu_hat is the MLE for mu
        """
        lmu     = self.log_prob(x)
        mu_hat  = self.get_mu_MLE(x)
        lmu_hat = self.clone().set_mu(mu_hat).log_prob(x)
        return -2 * (lmu - lmu_hat)

    def get_q_mu(self, x):
        """
        q_mu = -2 * lkhood(mu)/lkhood(mu_hat) if mu_hat <= mu else 0
        mu_hat is the MLE for mu
        """
        mu_hat  = self.get_mu_MLE(x)
        if mu_hat < self.mu:
            return 0
        lmu     = self.log_prob(x)
        lmu_hat = self.clone().set_mu(mu_hat).log_prob(x)
        return -2 * (lmu - lmu_hat)

    @staticmethod
    def get_distributions_for_statistics(sampling_model, testing_model, n_samples=1000, n_jobs=5, remove_pct_outliers=0.01):
        """
        samples histograms from this model(self), computes log_prob and t_mu/q_mu for each histogram
        other_model: the model with which log_prob and and t_mu/q_mu are computed

        returns an array with two cols. col0 is the log_prog, col1 is t_mu/q_mu
        """
        def f():
            x          = sampling_model.clone().rvs()
            mu_hat     = testing_model.get_mu_MLE(x)
            logprob_mu     = testing_model.log_prob(x)
            logprob_mu_hat = testing_model.clone().set_mu(mu_hat).log_prob(x)
            t_mu = -2 * (logprob_mu - logprob_mu_hat)
            q_mu = t_mu if mu_hat < testing_model.mu else 0 
            
            return logprob_mu, mu_hat, logprob_mu_hat, t_mu, q_mu

        k = np.r_[rlxutils.mParallel(n_jobs=n_jobs, verbose=30)(delayed(f)() for i in range(n_samples))]
        p1,p99 = np.percentile(k[:,3], [100*remove_pct_outliers,100*(1-remove_pct_outliers)])
        k = k[(k[:,1]>=p1)&(k[:,1]<=p99)]      
        
        return pd.DataFrame(k, columns=["logprob_mu", "mu_hat", "logprob_mu_hat", "t_mu", "q_mu"])

        
class SignalBg_ContinuousModel:
    def __init__(self, t, mu_s, sigma_s, n_events, mu):
        self.t, self.mu_s, self.sigma_s, self.n_events, self.mu = t, mu_s, sigma_s, n_events, mu
        self.init_distributions()
        
    def clone(self):
        return self.__class__(self.t, self.mu_s, self.sigma_s, self.n_events, self.mu)

    def set_mu(self, mu):
        self.mu = mu
        self.init_distributions()
        return self

    def init_distributions(self):
        self.b = stats.expon(scale=1/self.t)
        self.s = stats.norm(loc=self.mu_s, scale=self.sigma_s)     
        self.n_events = int(self.n_events)
        self.p_s, self.p_b = get_sgbg_probs(self.mu)
        self.s_tot = int(self.n_events*self.p_s)
        self.b_tot = int(self.n_events - self.s_tot)
        return self
        
    def rvs(self, n):     
        """
        sample from background and noise distributions and mix
        """   
        xb = self.b.rvs(self.b_tot)
        xs = self.s.rvs(self.s_tot)
        x = np.random.permutation(np.concatenate((xb,xs)))
        return x

    def bin_sample(self, x, bin_edges):
        return [np.sum((x>=bin_edges[i])&(x<bin_edges[i+1])) for i in range(len(bin_edges)-1)]
        
    def plot_sample(self, x, bins=30, density=True):

        # exclude small percentile larger to make nicer plot
        x = x[x<np.percentile(x, 99.5)]
        
        plt.hist(x, bins=bins, alpha=.5, density=density);

        if density:
            xr = np.linspace(np.min(x), np.max(x), 100).astype(np.float64)
            plt.plot(xr, np.exp(self.log_prob(xr)), color="red", label="analytical pdf")
        plt.legend(); plt.grid();
        plt.title(r"$t$=%.3f  ::  $\mu_s$=%.3f  ::  $\sigma_s$=%.3f  ::  $\mu$=%.3f"%(self.t, self.mu_s, self.sigma_s, self.mu))
        
    
    def log_prob(self, x):
        return np.log(self.p_s * gaus_pdf(x, self.mu_s, self.sigma_s) + self.p_b*exp_pdf(x, self.t) + 1e-100)
    
    def log_likelihood(self, x):
        return np.mean(self.log_prob(x))
        
            
            
