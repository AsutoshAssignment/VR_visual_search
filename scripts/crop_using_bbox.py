# scripts/crop_using_bbox.py

from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# =========================================================
# PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
CROPPED_DIR = ROOT_DIR / "data" / "cropped"

REPORT_DIR = ROOT_DIR / "report_assets" / "dataset"

ANNOTATION_FILE = PROCESSED_DIR / "merged_annotations.csv"

# Create crop folders
(CROPPED_DIR / "upper").mkdir(parents=True, exist_ok=True)
(CROPPED_DIR / "lower").mkdir(parents=True, exist_ok=True)
(CROPPED_DIR / "full").mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD ANNOTATIONS
# =========================================================

print("\nLoading merged annotations...")

df = pd.read_csv(ANNOTATION_FILE)

print(f"Total annotations loaded: {len(df)}")

# =========================================================
# HELPER FUNCTION
# =========================================================

def save_cropped_image(row):
    """
    Crop image using GT bounding box and save.
    """

    image_path = RAW_DIR / row["image_path"]

    clothing_label = row["clothing_label"]

    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    try:
        image = Image.open(image_path).convert("RGB")

        # Crop using bbox
        cropped = image.crop((x1, y1, x2, y2))

        # Preserve folder structure
        relative_path = Path(row["image_path"]).relative_to("img")

        save_path = CROPPED_DIR / clothing_label / relative_path

        save_path.parent.mkdir(parents=True, exist_ok=True)

        cropped.save(save_path)

        return True

    except Exception as e:
        print(f"Error processing {image_path}")
        print(e)

        return False

# =========================================================
# CROPPING LOOP
# =========================================================

print("\nStarting offline bbox cropping...")

success_count = 0
failure_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    success = save_cropped_image(row)

    if success:
        success_count += 1
    else:
        failure_count += 1

print("\n===================================")
print("CROPPING COMPLETE")
print("===================================")

print(f"Successful crops : {success_count}")
print(f"Failed crops     : {failure_count}")

# =========================================================
# GENERATE VISUALIZATION EXAMPLES
# =========================================================

print("\nGenerating bbox visualization examples...")

sample_rows = df.sample(6, random_state=42)

fig, axes = plt.subplots(6, 3, figsize=(12, 24))

for idx, (_, row) in enumerate(sample_rows.iterrows()):

    image_path = RAW_DIR / row["image_path"]

    clothing_label = row["clothing_label"]

    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    try:

        image = Image.open(image_path).convert("RGB")

        # -------------------------------
        # ORIGINAL IMAGE
        # -------------------------------

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis("off")

        # -------------------------------
        # BBOX VISUALIZATION
        # -------------------------------

        bbox_image = image.copy()

        draw = ImageDraw.Draw(bbox_image)

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3
        )

        axes[idx, 1].imshow(bbox_image)
        axes[idx, 1].set_title(f"BBox ({clothing_label})")
        axes[idx, 1].axis("off")

        # -------------------------------
        # CROPPED IMAGE
        # -------------------------------

        cropped = image.crop((x1, y1, x2, y2))

        axes[idx, 2].imshow(cropped)
        axes[idx, 2].set_title("Cropped")
        axes[idx, 2].axis("off")

    except Exception as e:

        print(f"Visualization error: {image_path}")
        print(e)

plt.tight_layout()

vis_save_path = REPORT_DIR / "bbox_crop_examples.png"

plt.savefig(vis_save_path)

plt.close()

print(f"\nSaved visualization examples -> {vis_save_path}")

print("\nAll cropping completed successfully.")