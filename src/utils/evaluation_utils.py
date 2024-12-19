import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px


def plot_target_organism_distribution(df):
    df = df[df['Target Source Organism According to Curator or DataSource'].notna()]
    # Count occurrences of each organism
    category_counts = df['Target Source Organism According to Curator or DataSource'].value_counts()
    category_counts = category_counts[category_counts.index != 'nan']
    category_counts = category_counts[:15]
    # Plot the pie chart
    plot = category_counts.plot.pie(figsize=(15, 15), autopct='%1.1f%%')
    
    # Display the plot
    import matplotlib.pyplot as plt
    plt.ylabel('')  # Optional: Remove y-axis label for better visualization
    plt.show()

def plot_target_organism_distribution_plotly(df):
    # Filter out rows with missing values in the specified column
    df = df[df['Target Source Organism According to Curator or DataSource'].notna()]
    
    # Count occurrences of each organism
    category_counts = df['Target Source Organism According to Curator or DataSource'].value_counts()
    category_counts = category_counts[category_counts.index != 'nan']
    category_counts = category_counts[:15]  # Select the top 15 categories

    # Convert the data to a DataFrame for Plotly
    category_data = category_counts.reset_index()
    category_data.columns = ['Organism', 'Count']

    # Create an interactive pie chart
    fig = px.pie(category_data, 
                 values='Count', 
                 names='Organism', 
                 title='Top 15 Target Source Organisms',
                 hole=0.4)  # Optional: Add a donut hole for style

    # Show the interactive plot
    fig.show()

def plot_metric_availability(df, metrics=["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"], ax=None):
    """
    Cleans the data by converting threshold values (e.g., '>50000', '<1') to NaN, calculates, 
    and plots the distribution of rows with defined values for each specified metric.
    
    Parameters:
    - df (pd.DataFrame): The BindingDB DataFrame containing the metrics.
    - metrics (list of str): List of metric columns to analyze.
    - ax (matplotlib.axes._subplots.AxesSubplot): Axis on which to plot, if provided.
    
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
    if ax is None:
        ax = plt.gca()
    ax.bar(availability_df["Metric"], availability_df["Percentage"], color="skyblue")
    ax.set_title("Availability of Each Metric in BindingDB")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Percentage of Rows with Defined Values")
    ax.set_ylim(0, 100)

def plot_metric_availability_with_plotly(df, metrics=["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]):
    """
    Cleans the data by converting threshold values (e.g., '>50000', '<1') to NaN, calculates, 
    and plots the distribution of rows with defined values for each specified metric using Plotly.
    
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
    
    # Create an interactive bar chart using Plotly
    fig = px.bar(
        availability_df,
        x="Metric",
        y="Percentage",
        title="Availability of Each Metric in BindingDB",
        labels={"Percentage": "Percentage of Rows with Defined Values", "Metric": "Metric"},
        text="Percentage",
        color="Metric",
    )
    
    # Update layout for better visualization
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        yaxis=dict(range=[0, 100], title="Percentage (%)"),
        xaxis=dict(title="Metric"),
        showlegend=False,
        width=800,
        height=500,
    )
    
    # Show the interactive plot
    fig.show()
    
    return availability_df
    

def plot_overlap_matrix(df, columns=['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)'], ax=None):
    ### let's see the overlap in the dataframe columns (metric columns are set as default)  

    # create an overlap matrix to compare the metrics
    overlap_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    # calculating pairwise overlaps as percentages
    for col1 in columns:
        for col2 in columns:
            # count non null values in both columns (overlap)
            overlap = df[col1].notnull() & df[col2].notnull()
            ## calculate overlap percentage relative to the average non-null counts in col1 and col2 for symmetry
            overlap_percentage = overlap.sum() / ((df[col1].notnull().sum() + df[col2].notnull().sum()) / 2) * 100
            overlap_matrix.loc[col1, col2] = overlap_percentage
            overlap_matrix.loc[col2, col1] = overlap_percentage  # ensure symmetry

    # create a string version of the overlap matrix for annotation with '%' symbols (otherwise gives an error)
    annot_matrix = overlap_matrix.map(lambda x: f"{x:.2f}%")

    # plot the overlap matrix with heatmap
    if ax is None:
        ax = plt.gca()
    sns.heatmap(overlap_matrix, annot=annot_matrix, fmt="", cmap="YlOrBr", cbar_kws={'label': 'Overlap Percentage (%)'}, ax=ax)
    ax.set_title("Pairwise Overlap Matrix of Ki, IC50, Kd, EC50")

def plot_overlap_matrix_with_plotly(df, columns=['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']):
    """
    Calculates and plots the overlap matrix between the specified columns using Plotly.
    
    Parameters:
    - df (pd.DataFrame): The BindingDB DataFrame containing the metrics.
    - columns (list of str): List of metric columns to analyze.
    
    Returns:
    - pd.DataFrame: The calculated overlap matrix as a DataFrame.
    """
    # Create an overlap matrix to compare the metrics
    overlap_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    # Calculate pairwise overlaps as percentages
    for col1 in columns:
        for col2 in columns:
            # Count non-null values in both columns (overlap)
            overlap = df[col1].notnull() & df[col2].notnull()
            # Calculate overlap percentage relative to the average non-null counts in col1 and col2 for symmetry
            overlap_percentage = overlap.sum() / ((df[col1].notnull().sum() + df[col2].notnull().sum()) / 2) * 100
            overlap_matrix.loc[col1, col2] = overlap_percentage
            overlap_matrix.loc[col2, col1] = overlap_percentage  # Ensure symmetry

    # Reset index for Plotly compatibility
    overlap_matrix = overlap_matrix.reset_index().melt(id_vars="index", var_name="Metric 1", value_name="Overlap (%)")
    overlap_matrix.rename(columns={"index": "Metric 2"}, inplace=True)

    # Create an interactive heatmap with Plotly
    fig = px.imshow(
        overlap_matrix.pivot(index="Metric 1", columns="Metric 2", values="Overlap (%)"),
        text_auto=".2f",
        color_continuous_scale="YlOrBr",
        title="Pairwise Overlap Matrix of Metrics",
        labels={"color": "Overlap Percentage (%)"},
    )

    # Customize layout for better appearance
    fig.update_layout(
        xaxis=dict(title="Metric 1"),
        yaxis=dict(title="Metric 2"),
        height=600,
        width=600,
    )
    
    # Show the figure
    fig.show()
    
    return overlap_matrix


def plot_organism_counts(df):
    ### let's see how STDs are distributed
    # Count the number of occurrences for each type of disease
    organism_counts = df['Target Source Organism According to Curator or DataSource'].value_counts().reset_index()
    organism_counts.columns = ['Target Source Organism', 'Count']

    plt.figure(figsize=(12, 12))

    ax = sns.barplot(
        x='Target Source Organism', 
        y='Count', 
        data=organism_counts
    )

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    plt.xticks(rotation=90, ha='right')
    plt.xlabel('Target Source Organism According to Curator or DataSource')
    plt.ylabel('Number of Rows')
    plt.title('Number of Rows per Target Source Organism')
    plt.tight_layout()
    plt.show()

def plot_ic50_boxplots(df, min_rows = 20):
    # filter to include only diseases with more than 20 rows (better visualisation)
    organism_counts = df['Target Source Organism According to Curator or DataSource'].value_counts()
    organisms_to_plot = organism_counts[organism_counts > min_rows].index
    filtered_df = df[df['Target Source Organism According to Curator or DataSource'].isin(organisms_to_plot)]
    
    plt.figure(figsize=(10, 10))
    
    ### boxplot of ic50 for every disease
    sns.boxplot( x='Target Source Organism According to Curator or DataSource', y='IC50 (nM)', data=filtered_df)
    
    plt.yscale('log')  # Set y-axis to log scale if the values vary significantly
    plt.xlabel('Type of STDs')
    plt.ylabel('IC50 (nM)')
    plt.xticks(rotation=90)
    plt.title('IC50 Distribution by disease')
    plt.tight_layout()
    plt.show()

def plot_most_targeted_proteins(df, organism = 'Human immunodeficiency virus 1', n = 20):
    df = df[df['Target Source Organism According to Curator or DataSource'] == organism]
    if organism == 'Human immunodeficiency virus 1':
        df['Target Name'] = df['Target Name'].str.replace("Dimer of ", "", regex=False)
        df['Target Name'] = df['Target Name'].str.replace("Reverse transcriptase protein", "Reverse transcriptase", regex=False)
    targeted = df['Target Name'].value_counts().head(n)
    plt.figure(figsize=(12, 8))
    sns.barplot( x=targeted.index, y=targeted.values, hue=targeted.index, palette='colorblind', dodge=False, legend=False)
    plt.xticks(rotation=90)
    plt.title(f'Most Targeted {organism} Proteins')
    plt.xlabel("Target Name")
    plt.ylabel("Count")
    plt.show()

def plot_most_targeted_proteins_plotly(df, organism='Human immunodeficiency virus 1', n=20):
    df = df[df['Target Source Organism According to Curator or DataSource'] == organism]
    
    if organism == 'Human immunodeficiency virus 1':
        df['Target Name'] = df['Target Name'].str.replace("Dimer of ", "", regex=False)
        df['Target Name'] = df['Target Name'].str.replace("Reverse transcriptase protein", "Reverse transcriptase", regex=False)
    
    targeted = df['Target Name'].value_counts().head(n).reset_index()
    targeted.columns = ['Target Name', 'Count']

    # Plot using Plotly
    fig = px.bar(targeted, x='Target Name', y='Count', color='Target Name', 
                 title=f'Most Targeted {organism} Proteins', 
                 labels={'Target Name': 'Target Name', 'Count': 'Count'},
                 color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(
        width=1000,  # Adjust the width here
        height=1000   # Optional: Adjust the height
    )
    fig.update_layout(xaxis_tickangle=90)
    fig.show()

def plot_publications_per_year(df):
    # Let's see the number of publications every year
    plt.figure(figsize=(12, 6))

    # Create a count plot for the publication year
    sns.countplot(x='year', data=df)

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Publication Year')
    plt.ylabel('Number of Articles Published')
    plt.title('Number of Articles Published per Year')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(dfa):
    #setting up the plot
    f = plt.figure(figsize=(19, 15))
    
    # Display the correlation matrix as a heatmap
    plt.matshow(dfa.corr(), fignum=f.number)
    #setting the plot (x ticks)
    plt.xticks(range(dfa.select_dtypes(['number']).shape[1]), dfa.select_dtypes(['number']).columns,fontsize=12, rotation=45)
    ##setting the plot (y ticks)
    plt.yticks(range(dfa.select_dtypes(['number']).shape[1]), dfa.select_dtypes(['number']).columns, fontsize=12)
    
    # adjust x axis tick parameters to add padding and display labels on the right
    plt.gca().xaxis.set_tick_params(pad=1, labelright=True)
    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


def plot_correlation_matrix2(dfa):
    # Compute the correlation matrix
    corr_matrix = dfa.corr()

    # Set up the plot
    f = plt.figure(figsize=(19, 15))

    # Display the correlation matrix as a heatmap
    plt.matshow(corr_matrix, fignum=f.number, cmap='coolwarm')
    
    # Setting the plot (x ticks)
    plt.xticks(
        range(corr_matrix.shape[1]), 
        corr_matrix.columns, 
        fontsize=12, 
        rotation=45,
        ha='left'  # Aligns the labels to the left
    )
    
    # Setting the plot (y ticks)
    plt.yticks(
        range(corr_matrix.shape[0]), 
        corr_matrix.columns, 
        fontsize=12
    )
    
    # Adjusting x-axis tick parameters
    plt.gca().xaxis.set_tick_params(pad=5, labelright=True)
    
    # Add color bar
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    
    # Set the title
    plt.title('Correlation Matrix', fontsize=16)
    
    # Show the plot
    plt.show()


def plot_sse(features_X, start=2, end=11):
    sse = []
    for k in range(start, end):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(features_X)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(sse.k, sse.sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")


def plot_pca_2d(df, clusters, kmc):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
    if kmc == True:
        plt.title("K-means Clustering Results")
    else:
        plt.title('Spectral Clustering results')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


def plot_pca_3d(df, clusters, kmc):
    pca3 = PCA(n_components=3)
    reduced_data = pca3.fit_transform(df)

    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2], c=clusters)
    if kmc==True:
        plt.title("3D PCA for K-means")
    else:
        plt.title("3D PCA for Spectral CLustering")    
    # show plot
    plt.show()

def plot_protein_target_heatmap(data, drug_class_col='Drug_Class', target_name_col='Target Name', title="Targeted HIV Proteins by Drug Class", figsize = (12,20)):
    """
    Plots a heatmap showing the count of protein targets by drug class.

    Parameters:
    - data (DataFrame): The DataFrame containing drug class and target name information.
    - drug_class_col (str): The column name for drug classes in the DataFrame.
    - target_name_col (str): The column name for protein target names in the DataFrame.
    - title (str): Title of the heatmap plot.
    
    Returns:
    - None: Displays the heatmap.
    """
    # Group the data by drug class and target name and count occurrences
    protein_target_counts = data.groupby([drug_class_col, target_name_col]).size().reset_index(name='Count')
    
    # Pivot the table for a better visualization
    table = protein_target_counts.pivot(index=drug_class_col, columns=target_name_col, values='Count').fillna(0)
    
    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(table, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.ylabel(drug_class_col)
    plt.xlabel(target_name_col)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_drug_distribution_by_year(data, year_col='year_y', drug_class_col='Drug_Class', title="Distribution of Drugs by Year (Side-by-Side by Drug Class)"):
    """
    Plots the distribution of drug classes by year using a side-by-side histogram.

    Parameters:
    - data (DataFrame): The DataFrame containing drug distribution data.
    - year_col (str): The column name for years in the DataFrame.
    - drug_class_col (str): The column name for drug classes in the DataFrame.
    - title (str): Title of the histogram plot.
    
    Returns:
    - None: Displays the histogram.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=year_col, hue=drug_class_col, multiple='dodge', kde=False)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Count of Drugs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_ic50_by_drug_class(data, drug_class_col='Drug_Class', ic50_col='IC50 (nM)', 
                            drug_class_order=None, title="Distribution of IC50 (nM) by Drug Class"):
    """
    Plots the distribution of IC50 values by drug class using a boxplot with a log-scaled and inverted y-axis.

    Parameters:
    - data (DataFrame): The DataFrame containing IC50 and drug class information.
    - drug_class_col (str): The column name for drug classes in the DataFrame.
    - ic50_col (str): The column name for IC50 values in the DataFrame.
    - drug_class_order (list): The specific order for drug classes in the plot.
    - title (str): Title of the boxplot.
    
    Returns:
    - None: Displays the boxplot.
    """
    # Set up the plotting style
    plt.figure(figsize=(12, 8))
    
    # Plot IC50 distribution by drug class using a boxplot
    sns.boxplot(data=data, x=drug_class_col, y=ic50_col, order=drug_class_order)
    
    # Apply a log scale to the y-axis for better visualization
    plt.yscale('log')
    
    # Labels and title
    plt.title(title)
    plt.xlabel("Drug Class")
    plt.ylabel("IC50 (nM)")
    plt.tight_layout()
    plt.show()


def group_similar_targets(hiv):
    """
    Groups similar protein targets in the HIV dataset by condensing overlapping or redundant names.

    Args:
        hiv (DataFrame): A pandas DataFrame containing a column named 'Target Name'.

    Returns:
        DataFrame: A new DataFrame with condensed target names.
    """
    # Create a copy of the dataset to avoid modifying the original
    hiv_condensed = hiv.copy()
    
    # Remove "Dimer of " from Target Names
    hiv_condensed['Target Name'] = hiv_condensed['Target Name'].str.replace("Dimer of ", "", regex=False)
    
    # Replace "Reverse transcriptase protein" with "Reverse transcriptase"
    hiv_condensed['Target Name'] = hiv_condensed['Target Name'].str.replace("Reverse transcriptase protein", "Reverse transcriptase", regex=False)
    
    # Condense "Gag-Pol polyprotein"
    hiv_condensed['Target Name'] = hiv_condensed['Target Name'].apply(lambda x: 'Gag-Pol polyprotein' if 'Gag-Pol' in x else x)
    
    # Condense "Reverse transcriptase" and related terms
    hiv_condensed['Target Name'] = hiv_condensed['Target Name'].apply(lambda x: 'Reverse transcriptase' if 'Reverse transcriptase' in x else x)

    # Condense "Protein Rev" and "Protein Rev [8-24]"
    hiv_condensed['Target Name'] = hiv_condensed['Target Name'].apply(lambda x: 'Protein Rev' if 'Protein Rev' in x else x)
    
    return hiv_condensed

def create_ic50_boxplot_plotly(dataframe):
    """
    Creates a boxplot for IC50 values against Target Name using Plotly.

    Args:
        dataframe (DataFrame): A pandas DataFrame containing 'Target Name' and 'IC50 (nM)' columns.

    Returns:
        Plotly Figure: A Plotly boxplot figure.
    """
    # Create the boxplot
    fig = px.box(dataframe, 
                 x='Target Name', 
                 y='IC50 (nM)', 
                 title='IC50 vs Target Name', 
                 labels={'Target Name': 'Target Name', 'IC50 (nM)': 'IC50 (nM)'},
                 color='Target Name')  # Color by target name

    # Update layout for log scale and x-axis label rotation
    fig.update_layout(
        yaxis_type="log",  # Apply log scale to the y-axis
        xaxis_tickangle=90,  # Rotate x-axis labels
        title="IC50 vs Target Name",
        width=1000,  # Adjust the width here
        height=1000   # Optional: Adjust the height
    )
    
    return fig
