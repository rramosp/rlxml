from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import sys
import progressbar
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KernelDensity


def plot_2Ddata(X, y, dots_alpha=.5, noticks=False):
    colors = cm.hsv(np.linspace(0, .7, len(np.unique(y))))
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X[y==label][:,0], X[y==label][:,1], color=colors[i], alpha=dots_alpha)
    if noticks:
        plt.xticks([])
        plt.yticks([])



def plot_2D_boundary(predict, mins, maxs, n=200, line_width=3, line_color="black", line_alpha=1, label=None):
    n = 200 if n is None else n
    mins -= np.abs(mins)*.2
    maxs += np.abs(maxs)*.2
    d0 = np.linspace(mins[0], maxs[0],n)
    d1 = np.linspace(mins[1], maxs[1],n)
    gd0,gd1 = np.meshgrid(d0,d1)
    D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))
    preds = predict(D)
    levels = np.sort(np.unique(preds))
    levels = [np.min(levels)-1] + [np.mean(levels[i:i+2]) for i in range(len(levels)-1)] + [np.max(levels)+1]
    p = (preds*1.).reshape((n,n))
    plt.contour(gd0,gd1,p, levels=levels, alpha=line_alpha, colors=line_color, linewidths=line_width)
    if label is not None:
        plt.plot([0,0],[0,0], lw=line_width, color=line_color, label=label)
    return np.sum(p==0)*1./n**2, np.sum(p==1)*1./n**2

def plot_2Ddata_with_boundary(predict, X, y, line_width=3, line_alpha=1, line_color="black", dots_alpha=.5, label=None, noticks=False):
    mins,maxs = np.min(X,axis=0), np.max(X,axis=0)    
    plot_2Ddata(X,y,dots_alpha)
    p0, p1 = plot_2D_boundary(predict, mins, maxs, line_width=line_width, 
                line_color=line_color, line_alpha=line_alpha, label=label )
    if noticks:
        plt.xticks([])
        plt.yticks([])
        
    return p0, p1

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


def ddistplot(x, plot_equivalent_gaussian=False, plot_equivalent_poisson=False, **kwargs):
    plt.plot(*kdensity_smoothed_histogram(x), color="black", lw=2, label="KDE data")
    plt.hist(x, density=True, **kwargs)
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

class KDClassifier:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X,y):
        """
        builds a kernel density estimator for each class
        """
        self.ndims = X.shape[1]
        self.kdes = {}
        for c in np.unique(y):
            stdx = np.min(np.std(X[y==c], axis=0))
            if "bandwidth" in self.kwargs.keys():
                bw = self.kwargs["bandwidth"]
                kwargs = {k:v for k,v in self.kwargs.items() if k!="bandwidth"}
            else:
                bw = 1.06*stdx*len(X)**-.2 if stdx != 0 else 1.
                kwargs = self.kwargs
            self.kdes[c] = KernelDensity(bandwidth=bw, **kwargs)
            self.kdes[c].fit(X[y==c])
            
        self.classes = list(self.kdes.keys())
        
        # build probability maps for each class
        n = int(np.power(1e5, 1/self.ndims))
        c = np.r_[[np.linspace(np.min(X[:,c]), np.max(X[:,c]),n) for c in range(X.shape[1])]]
        self.data_linspaces = c
        
        dV = np.product(c[:,1]-c[:,0])
        self.data_meshgrid = np.meshgrid(*c)
        c = np.r_[[i.flatten() for i in self.data_meshgrid]].T
        self.log_probmaps = [i.reshape( [n]*self.ndims) for i in self.kde_logprobs(c).T]
        self.probmaps = np.exp(self.log_probmaps)        
                
        # compute kl divergences between each pair of classes
        kldivs = np.zeros((len(self.classes), len(self.classes)))
        epsilon = 1e-50
        for c1 in range(0, len(self.classes)):
            for c2 in range(0, len(self.classes)):
                if c1==c2:
                    continue
                kldivs[c1,c2] = -dV*np.sum(self.probmaps[c1]*\
                                           (np.log(self.probmaps[c2]+epsilon)-np.log(self.probmaps[c1]+epsilon)))
        self.kldivs = kldivs        
        
        self.kldivs = pd.DataFrame(self.kldivs, index=self.classes, columns = self.classes)
        self.kldivs.index.name = "class"
        self.kldivs.columns.name = "KL divergence"
        
        return self

    
    def kde_logprobs(self, X):
        preds = []
        for i in sorted(self.classes):
            preds.append(self.kdes[i].score_samples(X))
        preds = np.array(preds).T # this is proba
        return preds
    
    def predict_proba(self, X):
        preds = self.kde_logprobs(X)
        preds = np.exp(preds)
        preds = preds/np.sum(preds,axis=1).reshape(-1,1)
        return preds

    def predict(self, X):
        """
        predicts the class with highest kernel density probability
        """
        preds = self.predict_proba(X)
        preds = preds.argmax(axis=1)
        preds = np.array([self.classes[i] for i in preds])

        return preds

    def score(self, X, y):
        return np.mean(y == self.predict(X))
        
    def plot_probmaps(self):

        if self.ndims==2:
            plt.figure(figsize=(5*len(self.classes), 3.5))
            for i in range(len(self.classes)):
                plt.subplot(1,len(self.classes), i+1)
                plt.contourf(*self.data_meshgrid, self.probmaps[i])
                plt.colorbar();
                plt.title("KDE probmap for class %d"%i)        
                
        elif self.ndims==1:
            for i in range(len(self.classes)):
                plt.plot(self.data_meshgrid[0], self.probmaps[i], label="class %d"%i)
            plt.title("KDE probaiblity maps")
            plt.grid();
            plt.legend();            
        
        else:
            raise ValueError("can only plot data with dims 1 or 2")



def distplot(x, pdf=None, pdf_name=None, fill_area=None, fill_area_label=None, legend_outside=True, **kwargs):
    if not 'alpha' in kwargs:
        kwargs['alpha'] = .5
    if not 'bins' in kwargs:
        kwargs['bins'] = 30
        
    counts = plt.hist(x, density=True, **kwargs)[0]
    
    if pdf is not None:
        xmin, xmax = np.min(x), np.max(x)
        xr = np.linspace(xmin, xmax, 100)
        plt.plot(xr, pdf(xr), color="black", label=pdf_name)
        if fill_area is not None: 
            fill_area[0] =  fill_area[0] if fill_area[0]!=-np.inf else np.min(x)
            fill_area[1] =  fill_area[1] if fill_area[1]!=np.inf else np.max(x)
            fill_area = np.linspace(*fill_area, 100)
            plt.fill_between(fill_area, pdf(fill_area), color="gray", alpha=1, label=fill_area_label)

    plt.ylim(0,np.max(counts))
    plt.grid("on");
    if pdf_name is not None or fill_area_label is not None:
        if legend_outside:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); 
        else:
            plt.legend();


class ConsecutiveSplit:
    
    def __init__(self, nrows_train, nrows_test, skip_last_train_rows=0):
        self.nrows_train = nrows_train
        self.nrows_test  = nrows_test
        self.skip_last_train_rows = skip_last_train_rows
        
    def split(self, X, y=None, groups=None):
        
        k = 0
        while k+self.nrows_train+self.nrows_test <= len(X):
            train_indexes = np.arange(k, k+self.nrows_train-self.skip_last_train_rows)
            test_indexes  = np.arange(k+self.nrows_train, k+self.nrows_train+self.nrows_test)
            yield train_indexes, test_indexes
            k += self.nrows_test


class TimedateContinuosCoverage_Ranger:
    """
    creates contiguous timedate ranges for train and for test.
    """
    def __init__(self, start_date, end_date,
                 train_period, test_period,
                 min_gap_period="0d", max_gap_period="0d",
                 verbose=False):

        bd_class = pd.tseries.offsets.BusinessDay

        self.start_date = start_date
        self.end_date = end_date
        self.train_period   = ru.to_timedelta(train_period)
        self.test_period    = ru.to_timedelta(test_period)
        self.min_gap_period = ru.to_timedelta(min_gap_period)
        self.max_gap_period = ru.to_timedelta(max_gap_period)
        self.verbose = verbose

    def __iter__(self):
        date = self.start_date
        while date + self.train_period + self.test_period < self.end_date:
            train_dates = (date, date + self.train_period)
            test_dates = (date + self.train_period, date + self.train_period + self.test_period)
            if self.verbose:
                print ("TR [", train_dates[0].strftime("%Y-%m-%d %H:%M:%S"), "-", train_dates[1].strftime(
                    "%Y-%m-%d %H:%M:%S"), "] -- TS [", \
                    test_dates[0].strftime("%Y-%m-%d %H:%M:%S"), "-", test_dates[1].strftime("%Y-%m-%d %H:%M:%S"), "]")
            date += self.test_period + self.min_gap_period + (
                                                             self.max_gap_period - self.min_gap_period) * np.random.random()

            yield train_dates, test_dates