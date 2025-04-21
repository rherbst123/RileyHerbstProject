import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

EXCLUDE_FIELDS = [
    "accessURI",
    "locality",
    "verbatimLocality",
    "verbatimCollectors",
    "verbatimCoordinates",
    "verbatimDateIdentified",
    "verbaitmEventDate",
    "originalMethod",
    "verbatimElevation",
    "verbatimIdentification",
    "habitat"
    
]

def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[.,]", "", regex=True)
            .replace({"nan": ""})
            .replace(
                {r"^(?:na|n/a|N/A|none|na\.?|n\.a\.?)$": ""},
                regex=True,
            )
        )
    return out

def collect_all_mismatches(transcribed_path: str,
                           ground_truth_path: str,
                           output_txt_path: str,
                           exclude: list[str] | None = None):

    exclude = set(exclude or EXCLUDE_FIELDS)
    read_opts = dict(dtype=str, keep_default_na=False)
    t_raw = pd.read_csv(transcribed_path, **read_opts)
    g_raw = pd.read_csv(ground_truth_path, **read_opts)

    t_df = clean_dataframe(t_raw)
    g_df = clean_dataframe(g_raw)

    catalog_order = t_raw["catalogNumber"].tolist()

    merged = pd.merge(
        t_df, g_df,
        on="catalogNumber",
        suffixes=("_t", "_g"),
        how="left"
    )

    fields = sorted((set(t_df.columns) & set(g_df.columns) - {"catalogNumber"}) - exclude)

    per_field_matches = {f: 0 for f in fields}
    total_matches = total_compares = 0
    mismatches_by_entry = {}

    for f in fields:
        t_col = merged[f"{f}_t"]
        g_col = merged[f"{f}_g"]
        equal_mask = t_col.eq(g_col)
        per_field_matches[f] += equal_mask.sum()
        total_matches += equal_mask.sum()
        total_compares += len(equal_mask)
        diff_rows = (~equal_mask).to_numpy().nonzero()[0]
        for idx in diff_rows:
            mismatches_by_entry.setdefault(merged["catalogNumber"][idx], []).append(
                {
                    "Field": f,
                    "Transcribed": t_col.iloc[idx],
                    "GroundTruth": g_col.iloc[idx],
                    "Similarity": round(string_similarity(t_col.iloc[idx], g_col.iloc[idx]), 3),
                }
            )

    with Path(output_txt_path).open("w", encoding="utf-8") as fh:
        for cat_num in catalog_order:
            if cat_num not in mismatches_by_entry:
                continue
            mistakes = mismatches_by_entry[cat_num]
            fh.write(f"Catalog Number: {cat_num}\n")
            fh.write(f"Total Mistakes: {len(mistakes)}\n")
            fh.write("=" * 60 + "\n")
            for rec in mistakes:
                fh.write(f"Field: {rec['Field']}\n")
                fh.write(f"  Transcribed:   {rec['Transcribed']}\n")
                fh.write(f"  Ground Truth:  {rec['GroundTruth']}\n")
                fh.write(f"  Similarity:    {rec['Similarity']}\n")
                fh.write("-" * 40 + "\n")
            fh.write("\n")

    field_accuracy = {
        f: round(100 * per_field_matches[f] / len(merged), 2) for f in fields
    }
    total_accuracy = round(100 * total_matches / total_compares, 2)
    return field_accuracy, total_accuracy

if __name__ == "__main__":
    field_acc, overall = collect_all_mismatches(r"C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\CSV\\Claude260_4_19_25.csv",
                   r"C:\\Users\\Riley\\Documents\\GitHub\\RileyHerbstProject\\FinishedPipeline\\Outputs\\GroundTruth\\260ImagesGroundTruth_Edit.csv", 
                   r"C:\\Users\\riley\\Desktop\\260Gpt_ClaudeConf_4_19_25.txt")
    print("Accuracy by field:")
    for k, v in sorted(field_acc.items(), key=lambda x: x[0].lower()):
        print(f"  {k}: {v:.2f}%")
    print(f"\nTOTAL accuracy across all fields: {overall:.2f}%")
