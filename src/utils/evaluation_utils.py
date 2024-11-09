import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_availability(df, metrics=["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]):
    """
    Cleans the data by converting threshold values (e.g., '>50000', '<1') to NaN, calculates, 
    and plots the distribution of rows with defined values for each specified metric.
    
    Parameters:
    - df (pd.DataFrame): The BindingDB DataFrame containing the metrics.
    - metrics (list of str): List of metric columns to analyze.
    
    Returns:
    - pd.DataFrame: A DataFrame with the count and percentage of rows with non-NaN values for each metric.
    """
    
    # Replace threshold values (e.g., '>50000', '<1') with NaN across the specified metrics
    df[metrics] = df[metrics].replace({r'^>.*$': np.nan, r'^<.*$': np.nan}, regex=True)
    
    # Convert each metric column to numeric, coercing errors (non-numeric values become NaN)
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Initialize a dictionary to store counts of non-NaN values for each metric
    metric_counts = {metric: df[metric].notna().sum() for metric in metrics}
    total_rows = len(df)
    
    # Create a DataFrame to show counts and percentages
    availability_df = pd.DataFrame({
        "Metric": metric_counts.keys(),
        "Count": metric_counts.values(),
        "Percentage": [(count / total_rows) * 100 for count in metric_counts.values()]
    })
    
    # Plot the availability distribution as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(availability_df["Metric"], availability_df["Percentage"], color="skyblue")
    plt.title("Availability of Each Metric in BindingDB")
    plt.xlabel("Metric")
    plt.ylabel("Percentage of Rows with Defined Values")
    plt.ylim(0, 100)
    plt.show()
    