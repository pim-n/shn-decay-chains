import numpy as np
from matplotlib import pyplot as plt

from _utils.termcolors import termcolors as tc

class Distribution:
    """
    Generates a PDF and CDF for any exponential decay using the half-life.

    Arguments:
        half_life   Half life of the isotope
        time_range  User-defined range for the distribution
        dt          Time interval for the time axis
        A           Amplitude in exponential decay equation
    
    Attributes:
        half_life   Half life of the isotope
        dt          Time interval for the decay graph
        time_range  User-defined range for the distribution
        exponential The exponential decay distribution
        pdf         The (normalised) Probability Density Function (PDF) of the exponential distribution
        cdf         The cumulative sum of the PDF
    """

    def __init__(self, half_life, time_range, time_start=0, A=1):
        self.half_life = half_life
        self.dt = 5*half_life/10_000
        self.time_range = np.arange(time_start, time_range, self.dt)

        self.exponential = A * np.exp(- np.log(2) * self.time_range/self.half_life) # exponential decay formula
        self.pdf = self.calculate_pdf(self.exponential, self.time_range)
        self.cdf = self.calculate_cdf(self.pdf)

    def calculate_pdf(self, exp, time_range):
        pdf = exp / np.trapz(exp, x=time_range) # normalize the distribution
        return pdf

    def calculate_cdf(self, pdf):
        cdf = np.cumsum(pdf) / np.sum(pdf) # cumulative sum
        return cdf

    def plot(self, function='all'):
        """
        Plot the exponential distribution ('exp'), Probability Density Function ('pdf'), Cumulative Density Function ('cdf'), or all functions.

        Keyword arguments:
        function    Choose between 'all', 'exp', 'pdf', 'cdf' (default 'all')
        """
        dists = {'exp': self.exponential, 'pdf': self.pdf, 'cdf': self.cdf}
        if function == 'all':
            for d in dists.keys():
                y = dists[d]
                plt.plot(self.time_range, y, label=d)
        elif function in dists.keys():
            y = dists[function]
            plt.plot(self.time_range, y, label=function)
        else:
            raise ValueError(tc.WARNING + "Invalid function. Leave function kwarg empty to plot all, or choose between 'exp', 'pdf' or 'cdf'." + tc.ENDC)

        plt.legend()
        plt.grid() 
        plt.show()