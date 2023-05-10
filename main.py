import pandas as pd


def read_csv(filename,ColName):
    """
    Read a csv file and perform data cleaning.

    Parameters:
    filename (str): The name of the csv file to be read

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
    df.set_index(["Country Name", "Year"],inplace=False)
    df = df.dropna()
    return df


df_CO2 = read_csv('CO2.csv','CO2')
df_GDP = read_csv('GDP.csv','GDP')


def merge_datasets(df1, df2,column):
    """
    Merges two dataframes by index, drops missing values and returns the merged dataframe.

    Args:
    df1: Pandas dataframe, the first dataframe to be merged.
    df2: Pandas dataframe, the second dataframe to be merged.

    Returns:
    Pandas dataframe, the merged dataframe with missing values dropped.
    """
    merged_df = pd.merge(df1, df2, on= column,  how='outer')
    merged_df.dropna(inplace=True)
    return merged_df


merge_df = merge_datasets(df_CO2,df_GDP,column=['Country Name','Year'])




import pandas as pd
from sklearn import preprocessing

x = merge_df[:,2:3].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)


def scaler(df):
    """ Expects a dataframe and normalises all
        columnsto the 0-1 range. It also returns
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max

arr, df_min, df_max = scaler(merge_df)

def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

np_arr =  backscale(arr, df_min, df_max)