"""
Final assignment - Z0 boson

This programme reads in, combines and filters data from two .csv files.
After outliers are removed, it produces a plot of data and performs a
minimised chi-squared fit to find the best values for the two parameters of the
theoretical fit. It then calculates the reduced chi-squared for the produced fit.

ID 10735793, created on 07/12/2021
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


M_Z_START = 90
GAMMA_Z_START = 3
H_BAR = 6.62607015*1e-34
FILENAME_1 = 'z_boson_data_1.csv'
FILENAME_2 = 'z_boson_data_2.csv'
DELIMITER = ','

def main():
    """
    This function is responsible for the flow of the code. It calls other
    functions in the right order to filter data, make calculations and
    produce a final plot representing the key results.
    """
    values = data(FILENAME_1, FILENAME_2)

    mass_width_array_initial = optimal_parameters(values)
    expected_values_curve = breit_wigner(values[:, 0], mass_width_array_initial[0],
                                         mass_width_array_initial[1])

    final_values = residuals_filter(values, expected_values_curve)
    mass_width_array = optimal_parameters(final_values)
    final_curve = breit_wigner(final_values[:, 0], mass_width_array[0], mass_width_array[1])
    chi_squared_value = chi_squared(final_curve, final_values[:, 1], final_values[:, 2])
    reduced_chi_squared = normalising_function(chi_squared_value, len(final_values),
                                               len(mass_width_array))

    pcov = curve_fit(breit_wigner, final_values[:, 0], final_values[:, 1],
                     p0=[mass_width_array[0], mass_width_array[1]], sigma=final_values[:, 2])[1]
    mass_width_errors = np.sqrt(np.diag(pcov))
    lifetime = lifetime_from_width(mass_width_array[1])
    lifetime_error = uncertainty(lifetime, mass_width_array[1], mass_width_errors[1])

    mass_width_lifetime = np.append(mass_width_array, lifetime)
    errors = np.append(mass_width_errors, lifetime_error)

    plot(final_values, final_curve, chi_squared_value,
         reduced_chi_squared, mass_width_lifetime, errors)

def data(datafile_1, datafile_2):
    """
    This function reads in the two data files as numpy arrays, combines them
    and checks for invalid data and outliers based on entrier and
    standard deviation values.
    Returns a sorted N-dimensional numpy array, filtered_data.
    """
    data_1 = read(datafile_1)
    data_2 = read(datafile_2)
    combined_file = np.vstack((data_1, data_2))
    filtered_data = drop_nan_values(combined_file)
    filtered_data = remove_zero_uncertainty_values(filtered_data)
    filtered_data = cross_section_filter(filtered_data)
    filtered_data = filtered_data[filtered_data[:, 0].argsort()]
    return filtered_data

def read(filename):
    """
    This function reads in a file and returns a numpy array
    corresponding to it.
    ----------
    Parameters: filename : .csv file, string
    Returns: numpy array
    """
    return np.genfromtxt(filename, delimiter=DELIMITER, comments='%')

def drop_nan_values(dataset):
    """
    Filters a dataset (numpy array) and returns a modified numpy array
    without any nan values.
    ----------
    Parameters: dataset : numpy array
    Returns: filtered_data : numpy array
    """
    dataset_pd = pd.DataFrame(dataset)
    filtered_data_pd = dataset_pd.dropna(axis=0, how='any')
    filtered_data = filtered_data_pd.to_numpy()
    return filtered_data

def remove_zero_uncertainty_values(dataset):
    """
    Filters data and only includes the measurements with non-zero uncertainty.
    This is important for the chi-squared fit to work.
    The new array only includes data from the measurements where
    the uncertainty had a value.
    ----------
    Parameters: dataset : numpy array
    Returns: new_dataset : numpy array
    """
    counter = 0
    new_dataset = np.zeros((0, 3))
    for entry in dataset[:, 2]:
        if entry != 0:
            new_dataset = np.vstack((new_dataset, dataset[counter]))
        counter += 1
    return new_dataset

def cross_section_filter(dataset):
    """
    Filters data based on the values of cross section. Returns a modified
    dataset where none of the values of the cross section are negative or
    over 3 sigmas (standard deviation) away from the mean of cross sections.
    ----------
    Parameters: dataset : numpy array
    Returns: new_dataset : numpy array
    """
    limit_3_sigma = 3 * np.std(dataset[:, 1])
    counter = 0
    new_dataset = np.empty((0, 3))
    for entry in dataset[:, 1]:
        if np.abs(entry - np.mean(dataset[:, 1])) <= limit_3_sigma and entry >= 0:
            new_dataset = np.vstack((new_dataset, dataset[counter]))
        counter += 1
    return new_dataset

def optimal_parameters(dataset):
    """
    Finds the parameters of a function with which the chi-square value will be
    the smallest possible.
    """
    lambda_function = lambda parameters: \
        chi_squared(breit_wigner(dataset[:, 0], parameters[0], parameters[1]),
                    dataset[:, 1], dataset[:, 2])
    results = fmin(lambda_function, (M_Z_START, GAMMA_Z_START))
    return results

def breit_wigner(energy, mass, width):
    """
    Calculates the value of the cross section.
    -------
    Parameters: E : float
                    energy in units of GeV
                m_Z : float
                    mass in units of Gev/c^2
                Gamma_Z : float
                    particle width in GeV
    Returns: sigma : cross section in nano-bars, float
    """
    gamma_ee = 0.08391
    numerator = 12*np.pi*energy**2*gamma_ee**2
    denominator = mass**2*((energy**2-mass**2)**2+mass**2*width**2)
    sigma_natural_units = numerator / denominator
    sigma = sigma_natural_units*0.3894*1e6
    return sigma

def residuals_filter(observed, predicted):
    """
    Calculates the residuals for each data point and removes the data points
    where the residual is larger than three times the uncertainty on the
    data point.
    ----------
    Parameters: observed : numpy array
                    original set of data
                predicted : numpy array
                    set of predicted values
    Returns: residual_dataset : numpy array
                the modified dataset
    """
    counter = 0
    residual = observed[:, 1] - predicted
    residual_dataset = np.zeros((0, 3))
    for entry in residual:
        if np.abs(entry) < 3*(observed[counter, 2]):
            residual_dataset = np.vstack((residual_dataset, observed[counter]))
        counter += 1
    return residual_dataset

def chi_squared(expected, observed, observed_uncertainty):
    """
    Calculates the chi-squared value of a dataset and a set of expected values.
    ----------
    Parameters: expected : numpy array
                    a set of values containing the theoretical predictions
                    for the results of an experiment
                observed : numpy array
                    contains the actual values measured during an experiment
                uncertainty : numpy array
                    uncertainty values for the measured values
    Returns: float
    """
    return np.sum(((expected - observed) / observed_uncertainty)**2)

def normalising_function(value, length_of_dataset, number_of_parameters):
    """
    Returns the normalised value of a function based on how many values were
    used to calculate the original value and the number of free parameters and
    the original value calculated without normalisation.
    ----------
    Parameters: value : float
                    not normalised value of the function
                length_of_dataset : int
                number_of_parameters : int
    Returns a float. This is the normalised value of the function.
    """
    return value / (length_of_dataset - number_of_parameters)

def lifetime_from_width(width):
    """
Calculates the lifetime of a particle in seconds.
    ----------
    Parameters: width : float
                    width of a particle in units of GeV
    Returns: lifetime : float

    """
    lifetime = H_BAR / (width*1e-9)
    return lifetime

def uncertainty(value_1, value_2, uncertainty_2):
    """
    Calculates the uncertainty of a value_1 in the case of products or quotients,
    applicable in the case where value_1 only depends on one variable with an
    uncertainty, value_2.
    Based on percentage uncertainties in the fractional uncertainty formula.
    The uncertainty returned and all parameters are floats.
    """
    uncertainty_1 = value_1 * (uncertainty_2/value_2)
    return uncertainty_1

def plot(file, line_fit, chi, reduced_chi, parameters, errors):
    """
    This function creates two plots.
    One is the cross section with error bars against energy, the other one is
    a plot of residual values.
    It also displays the numerical results of the code.
    Finally, it saves the produced figure.
    ----------
    Parameters: file: numpy array
                    contains x and y values with y value uncertainties
                line_fit: numpy array
                chi: float
                reduced_chi: float
                parameters: numpy array
                errors: numpy array
    """

    figure = plt.figure(figsize=(13, 12), facecolor='black', edgecolor='w')
    axes_main_plot = figure.add_subplot(211, facecolor='black')
    x_values_1 = file[:, 0]
    y_values_1 = file[:, 1]
    y_error_values_1 = file[:, 2]
    x_error_values_1 = 0

    axes_main_plot.scatter(x_values_1, y_values_1, marker='x', label='data', color='cyan')
    axes_main_plot.errorbar(x_values_1, y_values_1, yerr=y_error_values_1,
                            xerr=x_error_values_1, ls='none', color='springgreen',
                            alpha=0.8, capsize=2)
    axes_main_plot.plot(file[:, 0], line_fit, label='line fit')

    axes_main_plot.set_title('cross section ($\sigma$) plotted against energy (E)',
                             color='white', fontsize=15)
    axes_main_plot.set_xlabel('Energy (GeV)', color='white', fontsize=12)
    axes_main_plot.set_ylabel('Cross Section (nb)', color='white', fontsize=12)
    axes_main_plot.legend()
    axes_main_plot.grid(True, alpha=0.5)
    axes_main_plot.spines['top'].set_color('white')
    axes_main_plot.spines['bottom'].set_color('white')
    axes_main_plot.spines['left'].set_color('white')
    axes_main_plot.spines['right'].set_color('white')
    axes_main_plot.tick_params(colors='white', which='both')

    axes_main_plot.annotate(('mass = {:.2f}'.format(parameters[0])), (0, 0),
                            (0, -35), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='11', color='white')
    axes_main_plot.annotate(('± {:.2f} GeV/$c^2$'.format(errors[0])),
                            (0, 0), (80, -35), xycoords='axes fraction',
                            va='top', textcoords='offset points', fontsize='11',
                            color='white')
    axes_main_plot.annotate(('width = {:.3f}'.format(parameters[1])), (0, 0),
                            (0, -55), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='11', color='white')
    axes_main_plot.annotate(('± {:.3f} GeV'.format(errors[1])),
                            (0, 0), (83, -55), xycoords='axes fraction',
                            textcoords='offset points', va='top',
                            fontsize='11', color='white')
    axes_main_plot.annotate(('lifetime = {:.2e}'.format(parameters[2])), (0, 0), (0, -75),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='11', color='white')
    axes_main_plot.annotate(('± {:.2e} s'.format(errors[2])), (0, 0), (113, -75),
                            xycoords='axes fraction', textcoords='offset points',
                            va='top', fontsize='11', color='white')
    axes_main_plot.annotate((r'$\chi^2$ = {:.3f}'.
                             format(chi)), (1, 0), (-60, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='11', color='white')
    axes_main_plot.annotate((r'reduced $\chi^2$ = {:.3f}'.
                             format(reduced_chi)), (1, 0), (-104, -55),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='11', color='white')

    axes_residuals = figure.add_subplot(212, facecolor='black')
    residuals = file[:, 1] - line_fit
    axes_residuals.errorbar(file[:, 0], residuals, yerr=file[:, 2],
                            fmt='D', markersize=3.5, color='lightcoral', elinewidth=1.5)
    axes_residuals.plot(file[:, 0], 0 * file[:, 0], color='white', alpha=0.8)

    axes_residuals.set_title('Residuals', fontsize=15, color='white')
    axes_residuals.set_xlabel('Energy (GeV)', color='white', fontsize=12)
    axes_residuals.set_ylabel('Cross Section (nb)', color='white', fontsize=12)
    axes_residuals.grid(True, alpha=0.5)
    axes_residuals.spines['top'].set_color('white')
    axes_residuals.spines['bottom'].set_color('white')
    axes_residuals.spines['left'].set_color('white')
    axes_residuals.spines['right'].set_color('white')
    axes_residuals.tick_params(colors='white', which='both')

    figure.tight_layout()
    plt.savefig('cross_section_results_residuals.png',figsize=(13, 12), dpi=300,
                facecolor='black', edgecolor='white')
    plt.show()

if __name__ == '__main__':
    main()
