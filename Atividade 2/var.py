# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# UFBA - Stochastic Processes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Team: Bruno Oliveira, Caio Viturino e Lara Dourado
# Part of this code was obtained in https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from sklearn.preprocessing import StandardScaler
import scipy.stats

import warnings
warnings.filterwarnings("ignore")

class stochastic_process(object):
    def __init__(self, var):
        self.var = var
        self.number_of_bins = 48

        # Create the bins for each variable
        self.hist,self.bin_edges = np.histogram(self.var, bins=self.number_of_bins)

        # Legend configuration
        self.custom_lines = [Line2D([0], [0], linewidth=2, color='purple'),
                        Line2D([0], [0], linewidth=2, color='red')]
        # Set up list of candidate distributions to use
        self.dist_names = ['beta',
                      'expon',
                      'gamma',
                      'lognorm',
                      'norm',
                      'pearson3',
                      'triang',
                      'uniform',
                      'weibull_min',
                      'weibull_max']

        self.pdf_fitted = None

    def plot_fit(self, dist_names, var_num):
        # Plot
        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.suptitle('Processos estocásticos - Atividade (2)')

        # Variable 1 - Histogram
        var_hist = axs[0].hist(self.var, self.bin_edges, cumulative=False)
        axs[0].set_title('Histograma da Var ' + var_num)
        # Variable 1 - Boxplot
        axs[1].boxplot((self.var), vert=True, showmeans=True, meanline=True,
                   labels=('x'), patch_artist=True,
                   medianprops={'linewidth': 2, 'color': 'purple'},
                   meanprops={'linewidth': 2, 'color': 'red'})
        axs[1].set_title('Boxplot da Var ' + var_num)
        axs[1].legend(self.custom_lines, ['Median','Mean'], loc="upper right")
        # Variable 1 - Fit curve
        axs[2].hist(self.var, self.bin_edges, cumulative=False)
        axs[2].plot(var_hist[1], self.pdf_fitted, label=dist_names)
        axs[2].set_title('Fit da Var ' + var_num)
        axs[2].set_xlim(var_hist[1][0],var_hist[1][-1])
        axs[2].legend(loc="upper right")
        plt.show()

    def normalize(self, var):
        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        size = len(var)

        # Put all the elements in a column
        var_new = var.reshape(-1,1)
        scaler.fit(var_new)

        # Perform standardization by centering and scaling
        var_std = scaler.transform(var_new)
        var_std = var_std.flatten()
        return var_std

    def get_cum_observed_frequency(self, var_std):
        percentile_bins = np.linspace(0,100,51)
        # Get percentile cutoffs from the standardization of var
        percentile_cutoffs = np.percentile(var_std, percentile_bins)
        observed_frequency, _ = (np.histogram(var_std, bins=percentile_cutoffs))
        # Cumulative frequency
        cum_observed_frequency = np.cumsum(observed_frequency)
        return cum_observed_frequency, percentile_cutoffs

    def find_best_fit(self):
        # Set up empty lists to strore results
        chi_square = []
        p_values = []

        # Set up 50 bins for chi-square test
        percentile_bins = np.linspace(0, 100, 51)

        # Get each var Std
        var_std = self.normalize(self.var)

        cum_observed_frequency_var, percentile_cutoffs_var = self.get_cum_observed_frequency(var_std)

        # Loop through candidate distributions
        for distribution in self.dist_names:
            # Set up distribution and get fitted distribution parameters
            dist = getattr(scipy.stats, distribution)
            param = dist.fit(var_std)

            # Obtain the KS test P statistic, round it to 5 decimal places
            p = scipy.stats.kstest(var_std, distribution, args=param)[1]
            p = np.around(p, 5)
            p_values.append(p)

            # Get expected counts in percentile bins
            # This is based on a 'cumulative distrubution function' (cdf)
            cdf_fitted = dist.cdf(percentile_cutoffs_var, *param[:-2], loc=param[-2], scale=param[-1])
            expected_frequency = []
            for bin in range(len(percentile_bins)-1):
                expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
                expected_frequency.append(expected_cdf_area)

            # calculate chi-squared
            expected_frequency = np.array(expected_frequency) * len(self.var)
            cum_expected_frequency = np.cumsum(expected_frequency)
            ss = sum (((cum_expected_frequency - cum_observed_frequency_var) ** 2) / cum_observed_frequency_var)
            chi_square.append(ss)

        # Collate results and sort by goodness of fit (best at top)
        results = pd.DataFrame()
        results['Distribution'] = self.dist_names
        results['chi_square'] = chi_square
        results['p_value'] = p_values
        results.sort_values(['chi_square'], inplace=True)

        return results

    def fit_curve(self):
        results = self.find_best_fit()
        dist_names = results.to_numpy()

        # Get the probability distribution from scipy.stats
        dist = getattr(scipy.stats, dist_names[0][0])
        param = dist.fit(self.var)
        print("Parameters of the distributed fit curve: ")
        print("Mean: ", param[-2])
        print("Std: ", param[-1])

        var_hist = np.histogram(self.var, self.bin_edges)

        # Get line for each distribution (and scale to match observed data)
        self.pdf_fitted = dist.pdf(var_hist[1], *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz(var_hist[0], var_hist[1][:-1]) / np.trapz (self.pdf_fitted, var_hist[1])
        self.pdf_fitted *= scale_pdf
        return dist_names

def main():
    print("Digite o nome do arquivo .txt que deseja ler: ")
    nome = input()

    var = np.array([line.rstrip('\n') for line in open(str(nome) + '.txt')]).astype(np.float)
    best_fit_var = stochastic_process(var)
    dist_names = best_fit_var.fit_curve()
    print("Distribution name: ", dist_names[0][0])
    best_fit_var.plot_fit(dist_names[0][0], str(nome[-1]))

if __name__ == '__main__':
    main()
