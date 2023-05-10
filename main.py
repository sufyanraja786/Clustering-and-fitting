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

    return df


df_CO2 = read_csv('CO2.csv','CO2')
print(df_CO2)
