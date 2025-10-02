#!/usr/bin/env python3
"""
GVLiD Technical Validation Script
--------------------------------
Produces:
 - checksums.csv               : SHA-256 checksums for all images
 - image_quality_metrics.csv   : per-image width, height, filesize, brightness, sharpness, blurry_flag
 - metadata_completeness.csv   : completeness counts for metadata fields
 - curation_log.csv            : list of removed files (if any) and reasons
 - validation_report.json      : summary stats (brightness mean/std, sharpness mean, blurry %, 
                                 metadata completeness, class_counts, kappa values if annotations provided)
"""

import os
import json
import argparse
import hashlib
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import filters
from sklearn.metrics import cohen_kappa_score

# ----------------------- Helper functions -----------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def tenengrad_sharpness(gray_arr):
    gx = filters.sobel_h(gray_arr.astype("float32"))
    gy = filters.sobel_v(gray_arr.astype("float32"))
    fm = gx**2 + gy**2
    return float(np.mean(fm))

def mean_brightness(rgb_arr):
    r, g, b = rgb_arr[:,:,0], rgb_arr[:,:,1], rgb_arr[:,:,2]
    lum = 0.299*r + 0.587*g + 0.114*b
    return float(np.mean(lum))

def compute_checksums(image_paths, out_path, curation_log):
    checksums = []
    removed = []
    for p in tqdm(image_paths, desc="Checksums"):
        try:
            h = sha256_file(p)
            checksums.append({"filename": os.path.relpath(p), "sha256": h})
        except Exception as e:
            removed.append({"filename": os.path.relpath(p), "reason": str(e)})
    pd.DataFrame(checksums).to_csv(out_path, index=False)
    if removed:
        curation_log.extend(removed)

def compute_image_quality(image_paths, out_csv, blur_threshold=1e4):
    rows = []
    for p in tqdm(image_paths, desc="Image metrics"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.array(im)
                brightness = mean_brightness(arr)
                gray = np.array(im.convert("L"))
                sharpness = tenengrad_sharpness(gray)
                blurry_flag = 1 if sharpness < blur_threshold else 0
                rows.append({
                    "filename": os.path.relpath(p),
                    "width": im.width,
                    "height": im.height,
                    "filesize": os.path.getsize(p),
                    "brightness": brightness,
                    "sharpness": sharpness,
                    "blurry_flag": blurry_flag
                })
        except Exception as e:
            rows.append({"filename": os.path.relpath(p), "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def metadata_completeness(metadata_list):
    df = pd.json_normalize(metadata_list)
    stats = []
    total = len(df)
    for col in df.columns:
        nonnull = df[col].notna().sum()
        stats.append({
            "field": col,
            "non_null": int(nonnull),
            "total": int(total),
            "completeness_pct": float(nonnull/total*100)
        })
    return pd.DataFrame(stats)

def compute_kappa(labels_a, labels_b, boot=2000):
    kappa = cohen_kappa_score(labels_a, labels_b)
    rng = np.random.RandomState(42)
    boot_scores = []
    idx = np.arange(len(labels_a))
    for _ in range(boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        boot_scores.append(cohen_kappa_score(np.array(labels_a)[sample], np.array(labels_b)[sample]))
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return kappa, (lo, hi)

# ----------------------- Main -----------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    curation_log = []

    # Collect image paths (recursive)
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_paths = []
    for root, _, files in os.walk(args.dataset_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, f))
    print(f"Found {len(image_paths)} images under {args.dataset_dir}")

    # Graceful exit if no images found
    if len(image_paths) == 0:
        print("ERROR: No images found in the dataset directory!")
        print("➡ Please check that:")
        print("   - The path is correct: ", args.dataset_dir)
        print("   - The folder contains .jpg / .jpeg / .png / .bmp / .tif files")
        print("   - Use quotes if your path has spaces, e.g.:")
        print('     --dataset_dir "C:\\Users\\gayak\\Downloads\\GRAPEVINE_2809\\DATASET\\GVLiD"')
        return

    # Checksums
    compute_checksums(image_paths, os.path.join(args.out_dir, "checksums.csv"), curation_log)

    # Image quality
    df_metrics = compute_image_quality(image_paths, os.path.join(args.out_dir, "image_quality_metrics.csv"), args.blur_threshold)

    # Metadata completeness
    with open(args.metadata, "r") as f:
        metadata_json = json.load(f)
    metadata_list = metadata_json if isinstance(metadata_json, list) else metadata_json.get("data", [metadata_json])
    df_compl = metadata_completeness(metadata_list)
    df_compl.to_csv(os.path.join(args.out_dir, "metadata_completeness.csv"), index=False)

    # Annotation reliability
    kappa_results = {}
    if args.annotation_subset and os.path.exists(args.annotation_subset):
        ann = pd.read_csv(args.annotation_subset)
        if "Expert_A_Label" in ann and "Expert_B_Label" in ann:
            k, ci = compute_kappa(ann["Expert_A_Label"], ann["Expert_B_Label"])
            kappa_results["inter_rater"] = {"kappa": k, "ci": ci}
        if "Expert_A_Round2_Label" in ann:
            k, ci = compute_kappa(ann["Expert_A_Label"], ann["Expert_A_Round2_Label"])
            kappa_results["intra_rater"] = {"kappa": k, "ci": ci}

    # Summary report
    report = {
        "n_images": len(image_paths),
        "brightness_mean": float(df_metrics["brightness"].mean()),
        "brightness_std": float(df_metrics["brightness"].std()),
        "sharpness_mean": float(df_metrics["sharpness"].mean()),
        "sharpness_std": float(df_metrics["sharpness"].std()),
        "blurry_count": int(df_metrics["blurry_flag"].sum()),
        "blurry_pct": float(df_metrics["blurry_flag"].mean() * 100),
        "kappa_results": kappa_results
    }
    with open(os.path.join(args.out_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Validation complete. Report saved to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./GVLiD")
    parser.add_argument("--metadata", type=str, default="metadata.json")
    parser.add_argument("--out_dir", type=str, default="./docs")
    parser.add_argument("--annotation_subset", type=str, default="")
    parser.add_argument("--blur_threshold", type=float, default=1e4)
    args = parser.parse_args()
    main(args)
