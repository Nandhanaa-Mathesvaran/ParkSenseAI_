import json
import os

sets = ["train", "valid", "test"]

for split in sets:

    coco_path = f"dataset/{split}/_annotations.coco.json"
    label_folder = f"dataset/{split}/labels"

    os.makedirs(label_folder, exist_ok=True)

    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Store image information
    images = {}

    for img in coco["images"]:
        images[img["id"]] = (
            img["file_name"],
            img["width"],
            img["height"]
        )

    # Convert annotations
    for ann in coco["annotations"]:

        image_id = ann["image_id"]

        filename, width, height = images[image_id]

        x, y, w, h = ann["bbox"]

        # Get class id from COCO
        class_id = ann["category_id"]

        # Ignore generic "spaces" class
        if class_id == 0:
            continue

        # Convert classes
        # space-empty (1) -> 0
        # space-occupied (2) -> 1
        class_id = class_id - 1

        # COCO -> YOLO
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        box_width = w / width
        box_height = h / height

        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_folder, label_name)

        with open(label_path, "a") as f:
            f.write(
                f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
            )

    print(f"{split} converted successfully")

print("Conversion completed!")