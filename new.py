import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def read_csv(filename, ColName):
    """
    Read a csv file and perform data cleaning.

    Parameters:
    filename (str): The name of the csv file to be read
    ColName (str): The name of the column to be created in the cleaned DataFrame

    Returns:
    pandas.DataFrame: A cleaned pandas DataFrame
    """

    # Read csv file and skip first 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # Set Country Name as index
    df.set_index("Country Name", inplace=True)

    # Transpose columns from 1960 to 2021 and clean it
    df = df.loc[:, "1960":"2021"].stack().reset_index()
    df.columns = ["Country Name", "Year", ColName]
    df.set_index(["Country Name", "Year"], inplace=True)
    df = df.dropna()
    return df


def normalize_data(df):
    """
    Normalizes the data in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be normalized

    Returns:
    pandas.DataFrame: The normalized DataFrame
    """

    scaler = StandardScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_norm


def merge_datasets(df1, df2):
    """
    Merges two dataframes by index, drops missing values and returns the merged dataframe.

    Parameters:
    df1 (pandas.DataFrame): The first DataFrame to be merged
    df2 (pandas.DataFrame): The second DataFrame to be merged

    Returns:
    pandas.DataFrame: The merged DataFrame with missing values dropped
    """

    merged_df = pd.merge(df1, df2, on=["Country Name", "Year"], how="outer")
    merged_df.dropna(inplace=True)
    return merged_df


# Read in the CO2 and GDP data
df_CO2 = read_csv("CO2.csv", "CO2")
df_GDP = read_csv("GDP.csv", "GDP")

# Merge the CO2 and GDP data
df_merged = merge_datasets(df_CO2, df_GDP)

# Normalize the data
df_norm = normalize_data(df_merged)
print(df_norm)

# Cluster the data using KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(df_norm)

# Add the cluster labels to the original DataFrame
df_merged["Cluster"] = kmeans.labels_

# Plot the clusters
plt.scatter(df_merged["CO2"], df_merged["GDP"], c=df_merged["Cluster"])
plt.xlabel("CO2 Emissions (kt)")
plt.ylabel("GDP (current US$)")
plt.title("KMeans Clustering of CO2 Emissions and GDP")
plt.show()

# Print the results
print("Cluster Centers:")
print(kmeans)
# print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df_norm.columns))
print("\nCountry Counts by Cluster:")
print(df_merged["Cluster"].value_counts())

""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

import numpy as np


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []  # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper
