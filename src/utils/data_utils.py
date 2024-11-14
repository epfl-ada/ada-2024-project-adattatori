import pandas as pd
import numpy as np
import requests

def select_metric(df, metric = 'IC50 (nM)'):
    df[metric] = pd.to_numeric(df[metric], errors = 'coerce')
    return df.dropna(subset = metric)

def clean_na_columns(df, threshold = 0.5):
    threshold = len(df) * threshold

    for col in df.columns:
        try:
            # Attempt to convert the column to numeric (forcing np.float64)
            df[col] = pd.to_numeric(df[col], errors='raise').astype(np.float64)
        except (ValueError, TypeError):
            # If conversion fails, leave the column as it is
            pass
    
    return df.dropna(axis = 1, thresh = threshold)


def nan_to_numeric(df):
    # Select columns that contain NaNs
    columns_with_nans_nn = df.columns[df.isna().any()]

    # Convert columns with NaNs to np.float64 if they contain only numeric values or NaNs
    for col in columns_with_nans_nn:
        if df[col].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x)).all():
            # Convert column to np.float64
            df[col] = df[col].astype(np.float64)
    return df

# function to do an http request to get the publication year using CrossRef
def get_publication_year(doi):
    # format the URL to include the DOI in the request
    url = f"https://api.crossref.org/works/{doi}"
    try:
        ##send an HTTP GET request
        response = requests.get(url)
        #raise an error for bad responses
        response.raise_for_status()  
        ### parse the JSON response
        data = response.json()
        
        ### extract the publication year from the response data
        year = data['message']['published']['date-parts'][0][0]
        return year
    except Exception as e:
        ## print an error message if something goes wrong
        print(f"Could not retrieve year for DOI {doi}: {e}")
        return None