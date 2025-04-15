import pandas as pd
from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def clean_dataframe(df):
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == object:
            df_cleaned[col] = df_cleaned[col].str.replace(r'[.,]', '', regex=True)
    return df_cleaned

def collect_all_mismatches(transcribed_path, ground_truth_path, output_txt_path):
    # Load and clean data
    transcribed_df = pd.read_csv(transcribed_path)
    ground_truth_df = pd.read_csv(ground_truth_path)
    transcribed_clean = clean_dataframe(transcribed_df)
    ground_truth_clean = clean_dataframe(ground_truth_df)

    # Merge on catalogNumber
    merged_df = pd.merge(
        transcribed_clean, ground_truth_clean,
        on="catalogNumber", suffixes=("_transcribed", "_truth")
    )

    # Get intersecting fields to compare
    fields_to_compare = set(transcribed_clean.columns) & set(ground_truth_clean.columns)
    fields_to_compare.discard("catalogNumber")

 
    mismatches_by_entry = {}

    for col in sorted(fields_to_compare):
        transcribed_col = merged_df[f"{col}_transcribed"].fillna("").astype(str).str.strip()
        truth_col = merged_df[f"{col}_truth"].fillna("").astype(str).str.strip()

        for i in range(len(merged_df)):
            t_val = transcribed_col.iloc[i]
            g_val = truth_col.iloc[i]
            if t_val != g_val:
                catalog_number = merged_df["catalogNumber"].iloc[i]
                if catalog_number not in mismatches_by_entry:
                    mismatches_by_entry[catalog_number] = []
                mismatches_by_entry[catalog_number].append({
                    "Field": col,
                    "Transcribed": t_val,
                    "GroundTruth": g_val,
                    "Similarity": round(string_similarity(t_val, g_val), 3)
                })


    with open(output_txt_path, "w", encoding="utf-8") as f:
        for catalog_number, mismatches in mismatches_by_entry.items():
            f.write(f"Catalog Number: {catalog_number}\n")
            f.write(f"Total Mistakes: {len(mismatches)}\n")
            f.write("=" * 60 + "\n")
            for record in mismatches:
                f.write(f"Field: {record['Field']}\n")
                f.write(f"  Transcribed:   {record['Transcribed']}\n")
                f.write(f"  Ground Truth:  {record['GroundTruth']}\n")
                f.write(f"  Similarity:    {record['Similarity']}\n")
                f.write("-" * 40 + "\n")
            f.write("\n\n")


collect_all_mismatches(
    "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\11_Images_test.csv",
    "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\11_Images_GroundTruth.csv",
    "c:\\Users\\riley\\Desktop\\Mismatches_11_Images.txt"
)
