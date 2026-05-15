# scripts/preprocess_dataset.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

# =========================================================
# PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "report_assets" / "dataset"

IMG_DIR = RAW_DIR / "img"

EVAL_FILE = RAW_DIR / "list_eval_partition.txt"
BBOX_FILE = RAW_DIR / "list_bbox_inshop.txt"

# Create output folders
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def parse_eval_file(eval_file):
    """
    Parse list_eval_partition.txt
    """

    data = []

    with open(eval_file, "r") as f:
        lines = f.readlines()

    # Skip first two header lines
    lines = lines[2:]

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 3:
            continue

        image_path, item_id, split = parts

        data.append({
            "image_path": image_path,
            "item_id": item_id,
            "split": split
        })

    return pd.DataFrame(data)


def parse_bbox_file(bbox_file):
    """
    Parse list_bbox_inshop.txt
    """

    data = []

    with open(bbox_file, "r") as f:
        lines = f.readlines()

    # Skip first two header lines
    lines = lines[2:]

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 7:
            continue

        image_path = parts[0]
        clothing_type = int(parts[1])

        x1 = int(parts[3])
        y1 = int(parts[4])
        x2 = int(parts[5])
        y2 = int(parts[6])

        data.append({
            "image_path": image_path,
            "clothing_type": clothing_type,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return pd.DataFrame(data)


def map_clothing_type(x):
    mapping = {
        1: "upper",
        2: "lower",
        3: "full"
    }
    return mapping.get(x, "unknown")


# =========================================================
# MAIN PREPROCESSING
# =========================================================

print("\nParsing evaluation file...")
eval_df = parse_eval_file(EVAL_FILE)

print("Parsing bbox file...")
bbox_df = parse_bbox_file(BBOX_FILE)

print("Merging dataframes...")
merged_df = pd.merge(eval_df, bbox_df, on="image_path")

# Add readable clothing labels
merged_df["clothing_label"] = merged_df["clothing_type"].apply(map_clothing_type)

# =========================================================
# VALIDATE IMAGE PATHS
# =========================================================

print("Validating image paths...")

valid_rows = []

for idx, row in merged_df.iterrows():

    img_path = RAW_DIR / row["image_path"]

    if img_path.exists():
        valid_rows.append(row)

merged_df = pd.DataFrame(valid_rows)

print(f"Total valid images: {len(merged_df)}")

# =========================================================
# SAVE MASTER CSV
# =========================================================

master_csv = PROCESSED_DIR / "merged_annotations.csv"

merged_df.to_csv(master_csv, index=False)

print(f"Saved master annotations -> {master_csv}")

# =========================================================
# SAVE SPLIT CSVs
# =========================================================

train_df = merged_df[merged_df["split"] == "train"]
query_df = merged_df[merged_df["split"] == "query"]
gallery_df = merged_df[merged_df["split"] == "gallery"]

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
query_df.to_csv(PROCESSED_DIR / "query.csv", index=False)
gallery_df.to_csv(PROCESSED_DIR / "gallery.csv", index=False)

print("\nSaved split CSV files")

# =========================================================
# DATASET STATISTICS
# =========================================================

print("\nGenerating dataset statistics...")

split_counts = merged_df["split"].value_counts()

plt.figure(figsize=(6, 4))
split_counts.plot(kind="bar")

plt.title("Dataset Split Distribution")
plt.xlabel("Split")
plt.ylabel("Count")

plt.tight_layout()

split_plot_path = REPORT_DIR / "split_distribution.png"
plt.savefig(split_plot_path)
plt.close()

print(f"Saved -> {split_plot_path}")

# =========================================================
# CLOTHING TYPE DISTRIBUTION
# =========================================================

clothing_counts = merged_df["clothing_label"].value_counts()

plt.figure(figsize=(6, 6))
clothing_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")
plt.title("Clothing Type Distribution")

plt.tight_layout()

clothing_plot_path = REPORT_DIR / "clothing_distribution.png"
plt.savefig(clothing_plot_path)
plt.close()

print(f"Saved -> {clothing_plot_path}")

# =========================================================
# SAMPLE IMAGE VISUALIZATION
# =========================================================

print("\nGenerating sample image grid...")

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

categories = ["upper", "lower", "full"]

for row_idx, category in enumerate(categories):

    subset = merged_df[
        merged_df["clothing_label"] == category
    ]

    samples = subset.sample(
        min(3, len(subset)),
        random_state=42
    )

    for col_idx, (_, sample) in enumerate(samples.iterrows()):

        img_path = IMG_DIR / sample["image_path"]

        try:
            image = Image.open(img_path).convert("RGB")

            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].set_title(category)

        except Exception:
            axes[row_idx, col_idx].text(
                0.5,
                0.5,
                "Image Error",
                ha="center"
            )

        axes[row_idx, col_idx].axis("off")

plt.tight_layout()

sample_grid_path = REPORT_DIR / "sample_images.png"

plt.savefig(sample_grid_path)
plt.close()

print(f"Saved -> {sample_grid_path}")

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n===================================")
print("DATASET PREPROCESSING COMPLETE")
print("===================================")

print("\nDataset Split Sizes:")
print(f"Train Images   : {len(train_df)}")
print(f"Query Images   : {len(query_df)}")
print(f"Gallery Images : {len(gallery_df)}")
print(f"Total Images   : {len(merged_df)}")

print("\nClothing Type Distribution:")
print(clothing_counts)

print("\nSaved Files:")
print(f"Train CSV      : {PROCESSED_DIR / 'train.csv'}")
print(f"Query CSV      : {PROCESSED_DIR / 'query.csv'}")
print(f"Gallery CSV    : {PROCESSED_DIR / 'gallery.csv'}")
print(f"Merged CSV     : {master_csv}")

print("\nReport Assets:")
print(f"Split Plot     : {split_plot_path}")
print(f"Clothing Plot  : {clothing_plot_path}")
print(f"Sample Grid    : {sample_grid_path}")

print("\nAll preprocessing completed successfully.")

# =========================================================
# SAVE DATASET STATISTICS
# =========================================================

stats_df = pd.DataFrame({
    "Metric": [
        "Train Images",
        "Query Images",
        "Gallery Images",
        "Total Images",
        "Upper Body",
        "Lower Body",
        "Full Body"
    ],
    "Count": [
        len(train_df),
        len(query_df),
        len(gallery_df),
        len(merged_df),
        clothing_counts.get("upper", 0),
        clothing_counts.get("lower", 0),
        clothing_counts.get("full", 0)
    ]
})

stats_csv_path = REPORT_DIR / "dataset_statistics.csv"

stats_df.to_csv(stats_csv_path, index=False)

print(f"Saved dataset statistics -> {stats_csv_path}")