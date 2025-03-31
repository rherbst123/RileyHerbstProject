import pandas as pd
import numpy as np
from scipy.stats import entropy

# Function to compute KL divergence for two discrete distributions.
def compute_kl_divergence(p, q):
    # Convert inputs to float arrays and add a small constant to avoid division by zero.
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    
    # Normalize to create proper probability distributions.
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

# Read the CSV files.
df1 = pd.read_csv("C:\\Users\\riley\\Desktop\\Code\\Outputs\\Woop2.csv")
df2 = pd.read_csv("C:\\Users\\riley\\Desktop\\Code\\Outputs\\Woop3.csv")

# Create a union of all columns from both dataframes.
all_columns = list(set(df1.columns).union(set(df2.columns)))

# Reindex both dataframes so that they have all the columns.
# Missing columns will be filled with NaN.
df1 = df1.reindex(columns=all_columns)
df2 = df2.reindex(columns=all_columns)

# Dictionary to store KL divergence for each field.
results = {}

# Loop through each column and compute distributions.
for col in all_columns:
    # If the column is numeric in both dataframes, compute a histogram distribution.
    if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
        # Drop missing values and combine the data to decide on common bin edges.
        combined = np.concatenate([df1[col].dropna().values, df2[col].dropna().values])
        bins = np.histogram_bin_edges(combined, bins=10)
        p_hist, _ = np.histogram(df1[col].dropna(), bins=bins)
        q_hist, _ = np.histogram(df2[col].dropna(), bins=bins)
    else:
        # For non-numeric (categorical) data, use the union of all observed values.
        union_values = pd.Index(df1[col].dropna().unique()).union(pd.Index(df2[col].dropna().unique()))
        p_counts = df1[col].value_counts().reindex(union_values, fill_value=0)
        q_counts = df2[col].value_counts().reindex(union_values, fill_value=0)
        p_hist = p_counts.values
        q_hist = q_counts.values
    
    # Compute the KL divergence for the current column.
    kl_div = compute_kl_divergence(p_hist, q_hist)
    results[col] = kl_div

# Print out the KL divergence for each field.
print("KL Divergence for each field:")
for col, kl in results.items():
    print(f"{col}: {kl}")
