import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# import the dataset
df = pd.read_csv("normalised.csv")
def fit_poly(x, y, degree):
    """
    Fit a polynomial function to the data.

    Args:
    x: independent variable data
    y: dependent variable data
    degree: degree of the polynomial function to fit

    Returns:
    popt: coefficients of the polynomial fit
    pcov: covariance matrix of the fit
    """
    popt, pcov = curve_fit(lambda x, *p: np.polyval(p, x), x, y, p0=np.ones(degree + 1))
    return popt, pcov
# create a subset of the data with only CO2 and GDP columns
df_subset = df[['CO2', 'GDP']]

# define the independent and dependent variables
x = df_subset['GDP']
y = df_subset['CO2']

# fit a polynomial function of degree 2 to the data
popt, pcov = fit_poly(x, y, degree=2)

# plot the original data and the fitted function
plt.scatter(x, y, label='Data')
plt.plot(x, np.polyval(popt, x), 'r-', label='Fit')
plt.xlabel('GDP')
plt.ylabel('CO2')
plt.legend()
plt.show()
# predict CO2 level in 2030
future_year = 2030
predicted_gdp = df_subset['GDP'].max() + 0.1  # assume 10% GDP growth
predicted_co2 = np.polyval(popt, predicted_gdp)

# print the predicted CO2 level
print(f"Predicted CO2 level in {future_year}: {predicted_co2:.2f} metric tons per capita")
