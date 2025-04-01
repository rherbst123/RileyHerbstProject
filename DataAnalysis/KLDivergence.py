import pandas as pd
import numpy as np
from scipy.stats import entropy

def compute_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    
    # Normalize to create proper probability distributions.
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

df1 = pd.read_csv("C:\\Users\\riley\\Desktop\\KLDivergence\\TranscriptionGT.csv") # Ground Truth to compare to
df2 = pd.read_csv("C:\\Users\\riley\\Desktop\\KLDivergence\\Transcription_Post_Seg.csv") # Pre or post seg

# Create a consistently ordered list of columns.
all_columns = sorted(list(set(df1.columns).union(set(df2.columns))))

df1 = df1.reindex(columns=all_columns)
df2 = df2.reindex(columns=all_columns)

results = {}

for col in all_columns:
    if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
        combined = np.concatenate([df1[col].dropna().values, df2[col].dropna().values])
        bins = np.histogram_bin_edges(combined, bins=10)
        p_hist, _ = np.histogram(df1[col].dropna(), bins=bins)
        q_hist, _ = np.histogram(df2[col].dropna(), bins=bins)
    else:
        union_values = pd.Index(df1[col].dropna().unique()).union(pd.Index(df2[col].dropna().unique()))
        p_counts = df1[col].value_counts().reindex(union_values, fill_value=0)
        q_counts = df2[col].value_counts().reindex(union_values, fill_value=0)
        p_hist = p_counts.values
        q_hist = q_counts.values

    kl_div = compute_kl_divergence(p_hist, q_hist)
    results[col] = kl_div

print("KL Divergence for each field:")
for col in all_columns:
    print(f"{col}: {results[col]}")
