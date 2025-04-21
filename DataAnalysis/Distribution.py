
import pandas as pd
import matplotlib.pyplot as plt
import difflib
from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def clean_dataframe(df):
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == object:
            df_cleaned[col] = df_cleaned[col].str.replace(r'[.,]', '', regex=True)
    return df_cleaned

def count_entry_mismatches(row, fields):
    count = 0
    for field in fields:
        val_t = str(row[f"{field}_transcribed"]).strip()
        val_g = str(row[f"{field}_truth"]).strip()
        if val_t != val_g:
            count += 1
    return count

def analyze_mismatches(transcribed_path, ground_truth_path, output_txt_path, show_chart=True):
    # Load and clean data
    transcribed_df = pd.read_csv(transcribed_path)
    ground_truth_df = pd.read_csv(ground_truth_path)
    transcribed_clean = clean_dataframe(transcribed_df)
    ground_truth_clean = clean_dataframe(ground_truth_df)

    # Merge on catalogNumber
    merged_clean_df = pd.merge(
        transcribed_clean, ground_truth_clean,
        on="catalogNumber", suffixes=("_transcribed", "_truth")
    )

    # Columns to compare
    columns_to_compare = set(transcribed_clean.columns) & set(ground_truth_clean.columns)
    columns_to_compare.discard("catalogNumber")

    # Count mismatches per row
    merged_clean_df["mismatch_count"] = merged_clean_df.apply(
        count_entry_mismatches, axis=1, fields=columns_to_compare
    )

    # Create mismatch distribution
    mismatch_distribution = merged_clean_df["mismatch_count"].value_counts().sort_index()
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for mistakes, count in mismatch_distribution.items():
            f.write(f"Entries with {mistakes} mistake(s): {count}\n")

     # Bar chart
        if show_chart:
            plt.figure(figsize=(10, 6))
            plt.bar(mismatch_distribution.index, mismatch_distribution.values)
            plt.title("Number of Entries by Mistake Count")
            plt.xlabel("Number of Mistakes per Entry")
            plt.ylabel("Number of Entries")
            plt.xticks(mismatch_distribution.index)
            max_y = mismatch_distribution.values.max()
            plt.yticks(range(0, max_y + 5, 5))
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()


analyze_mismatches("C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\Claude260_4_19_25.csv",
                   "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\GroundTruth\\260ImagesGroundTruth_Edit.csv", 
                   "C:\\Users\\riley\\Desktop\\260Gpt_ClaudeConf_4_19_25.txt")

# analyze_mismatches("C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\260Images_4_10_25_Cropped_Full_Images_GPT4o.csv",
#                    "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\260ImagesGroundTruth_4_9_25_GPT4o.csv", 
#                    "C:\\Users\\riley\\Desktop\\260Images.txt")

# analyze_mismatches("C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\GPT_Corrections_4_16_25_260Run.csv",
#     "C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\260ImagesGroundTruth_4_9_25_GPT4o.csv", 
#                    "C:\\Users\\riley\\Desktop\\CorrectedTest_Test.txt")
 