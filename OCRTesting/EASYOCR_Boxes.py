import easyocr
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def get_rect_coords(box):
    """
    Convert EasyOCR's quadrilateral box into (xmin, ymin, xmax, ymax).
    box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)

def merge_clusters(boxes, labels):
    """
    Given bounding boxes and their cluster labels, create one bounding rectangle per cluster.
    :param boxes: list of (xmin, ymin, xmax, ymax)
    :param labels: cluster assignment for each box
    :return: list of merged boxes in 4-point format (like EasyOCR: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    """
    merged = []
    unique_labels = set(labels)
    if -1 in unique_labels:
        # -1 is the "noise" cluster in DBSCAN; we typically ignore or handle it specially
        unique_labels.remove(-1)
    
    for cluster_id in unique_labels:
        # gather boxes belonging to cluster_id
        cluster_indices = [i for i, lab in enumerate(labels) if lab == cluster_id]
        xs, ys = [], []
        for i in cluster_indices:
            (xmin, ymin, xmax, ymax) = boxes[i]
            xs.extend([xmin, xmax])
            ys.extend([ymin, ymax])
        
        # build one bounding rectangle
        cluster_xmin, cluster_xmax = min(xs), max(xs)
        cluster_ymin, cluster_ymax = min(ys), max(ys)
        # convert back to four-point format
        merged_box = [
            [cluster_xmin, cluster_ymin],
            [cluster_xmax, cluster_ymin],
            [cluster_xmax, cluster_ymax],
            [cluster_xmin, cluster_ymax]
        ]
        merged.append(merged_box)
    return merged

# 1) Run EasyOCR
reader = easyocr.Reader(['en'])  # Example: English
img_path = 'OCRTesting\\TestImages\\TestImage_1.jpg'
results = reader.readtext(img_path)

# 2) Convert bounding boxes to a simpler rect form + collect center points
rects = []
centers = []
for (quad_box, text, confidence) in results:
    (xmin, ymin, xmax, ymax) = get_rect_coords(quad_box)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    rects.append((xmin, ymin, xmax, ymax))
    centers.append([cx, cy])

# 3) Cluster the center points using DBSCAN
# Adjust eps to control how close bounding boxes must be to cluster
# min_samples=1 means single box can form a cluster by itself
X = np.array(centers)
db = DBSCAN(eps=50, min_samples=1).fit(X)  # <-- Tweak eps based on typical spacing
labels = db.labels_

# 4) Merge bounding boxes in each cluster
merged_boxes = merge_clusters(rects, labels)

# 5) Draw the merged bounding boxes
image = cv2.imread(img_path)
for box in merged_boxes:
    coords = np.array(box, dtype=np.int32)
    cv2.polylines(image, [coords], isClosed=True, color=(0,0,255), thickness=2)

out_file = 'OCRTesting\\TestImage_Results\\result4.jpg'
cv2.imwrite(out_file, image)
print(f"Saved clustered bounding-box image to {out_file}")
