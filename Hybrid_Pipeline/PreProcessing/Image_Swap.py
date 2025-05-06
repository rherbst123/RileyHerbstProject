import os
import shutil

SOURCE_DIR   = r"C:\Users\Riley\Desktop\300ImageTess\300Images_4_9_25_Resized"          
PARENT_DIR   = r"C:\Users\Riley\Desktop\TextCorrection\260ImagesSegmentted4_19_25_FourthRun"             

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def build_source_map(src_root):
    src_map = {}
    for fname in os.listdir(src_root):
        root, ext = os.path.splitext(fname)
        if ext.lower() in EXTENSIONS:
            src_map[root] = os.path.join(src_root, fname)
    return src_map

def replace_visualizations(src_root, parent_root):
    src_map = build_source_map(src_root)
    print(f"Found {len(src_map)} candidate replacement images in SOURCE_DIR")


    for base_id in os.listdir(parent_root):
        images_dir = os.path.join(parent_root, base_id, "images")
        if not os.path.isdir(images_dir):
            continue                                  

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