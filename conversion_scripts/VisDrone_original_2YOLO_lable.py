import os
import pandas as pd
from PIL import Image

BASE = "../datasets/NPS/NPSvisdroneStyle"
VISIMG_BASE = "../datasets/NPS/AllFrames"

def convert(bbox, img_size):
    # Convert VisDrone bbox to YOLO format
    # Input:  top_left_x, top_left_y, width, height (pixels)
    # Output: x_center, y_center, width, height (relative 0-1)
    dw = 1 / img_size[0]
    dh = 1 / img_size[1]
    x = (bbox[0] + bbox[2] / 2) * dw   # center x
    y = (bbox[1] + bbox[3] / 2) * dh   # center y
    w = bbox[2] * dw                    # relative width
    h = bbox[3] * dh                    # relative height
    return (x, y, w, h)

def get_image_name(annotation_path, image_dir):
    # Clip_1_00000.txt → find Clip_1_00001.png
    clip_id, frameid = os.path.basename(annotation_path).split("_")[1:3]
    clip_id  = int(clip_id.strip())
    frameid  = int(frameid.split(".")[0].strip())
    return os.path.join(image_dir, f"Clip_{clip_id}_{str(frameid+1).zfill(5)}.png")

def ChangeToYolo5(split):
    YOLO_LABELS_PATH = os.path.join(BASE, split, "labels")
    VISANN_PATH      = os.path.join(BASE, split, "annotations")
    VISIMG_PATH      = os.path.join(VISIMG_BASE, split)

    print(f"\n--- Processing split: {split} ---")
    print(f"  Annotations: {VISANN_PATH}")
    print(f"  Images:      {VISIMG_PATH}")
    print(f"  YOLO labels: {YOLO_LABELS_PATH}")

    if not os.path.exists(YOLO_LABELS_PATH):
        os.makedirs(YOLO_LABELS_PATH)

    ann_files = os.listdir(VISANN_PATH)
    print(f"  Found {len(ann_files)} annotation files")

    for file in ann_files:
        image_path = get_image_name(file, VISIMG_PATH)
        ann_file   = os.path.join(VISANN_PATH, file)
        out_path   = os.path.join(YOLO_LABELS_PATH, file)

        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"  WARNING: Image not found: {image_path}")
            continue

        out_file = open(out_path, 'w')
        try:
            bbox_data = pd.read_csv(ann_file, header=None).values
            img       = Image.open(image_path)
            img_size  = img.size  # (width, height)

            for row in bbox_data:
                # row[4]=score(1=valid), row[5]=category(1=drone)
                if row[4] == 1 and 0 < row[5] < 11:
                    label = convert(row[:4], img_size)
                    out_file.write(
                        str(row[5] - 1) + " " +
                        " ".join(f'{x:.6f}' for x in label) + '\n'
                    )
        except Exception as e:
            print(f"  ERROR processing {file}: {e}")
        finally:
            out_file.close()

    print(f"  {split} done!")

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        ChangeToYolo5(split)
    print("\nAll splits done!")
