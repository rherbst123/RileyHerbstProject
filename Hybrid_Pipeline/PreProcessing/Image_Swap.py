import os
import shutil

# ===‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 1️⃣  UPDATE THESE TWO PATHS FOR YOUR MACHINE
# ----------------------------------------------------
SOURCE_DIR   = r"C:\Users\Riley\Desktop\300ImageTess\300Images_4_14_25_SatBri_Completed"          # e.g.  ...\NewImages
PARENT_DIR   = r"C:\Users\Riley\Desktop\TextCorrection\260Test"             # e.g.  ...\ParentFolder
# ----------------------------------------------------
# extensions we accept for both source & target
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def build_source_map(src_root):
    """
    Return dict {base_id: full_source_path}.
    Example: 0002_V0576008F → C:\...\Input10Images\0002_V0576008F.jpg
    """
    src_map = {}
    for fname in os.listdir(src_root):
        root, ext = os.path.splitext(fname)
        if ext.lower() in EXTENSIONS:
            src_map[root] = os.path.join(src_root, fname)
    return src_map

def replace_visualizations(src_root, parent_root):
    src_map = build_source_map(src_root)
    print(f"Found {len(src_map)} candidate replacement images in SOURCE_DIR")

    # every sub‑folder of PARENT_DIR is one specimen ID
    for base_id in os.listdir(parent_root):
        images_dir = os.path.join(parent_root, base_id, "images")
        if not os.path.isdir(images_dir):
            continue                                  # skip stray files

        src_path = src_map.get(base_id)
        if not src_path:
            print(f"[skip] {base_id}: no matching file in SOURCE_DIR")
            continue

        # look for any file whose name ends in _segmentation_visualization
        replaced_any = False
        for tgt_fname in os.listdir(images_dir):
            name_no_ext, ext = os.path.splitext(tgt_fname)
            if ext.lower() not in EXTENSIONS:
                continue
            if not name_no_ext.lower().endswith("_segmentation_visualization"):
                continue

            tgt_path = os.path.join(images_dir, tgt_fname)
            try:
                shutil.copy2(src_path, tgt_path)      # overwrite in‑place
                print(f"[OK]  {base_id}: replaced {tgt_fname}")
                replaced_any = True
            except Exception as e:
                print(f"[ERR] {base_id}: {e}")

        if not replaced_any:
            print(f"[skip] {base_id}: no *_segmentation_visualization file")

if __name__ == "__main__":
    replace_visualizations(SOURCE_DIR, PARENT_DIR)