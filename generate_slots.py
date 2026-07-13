import json


annotation_file = "dataset/train/_annotations.coco.json"


target_image = "2012-09-11_15_16_58_jpg.rf.61d961a86c9a16694403dfcb72cd450c.jpg"


with open(annotation_file, "r") as f:
    data = json.load(f)


# Find image id
image_id = None

for img in data["images"]:

    if img["file_name"] == target_image:
        image_id = img["id"]
        break


if image_id is None:
    print("Image not found")
    exit()


slots = []


# Extract only that image slots
for annotation in data["annotations"]:

    if annotation["image_id"] == image_id:

        x, y, w, h = annotation["bbox"]

        slots.append([
            [int(x), int(y)],
            [int(x+w), int(y+h)]
        ])


with open("auto_slots.json", "w") as f:
    json.dump(slots, f, indent=4)


print("Image:", target_image)
print("Total parking slots:", len(slots))
print("auto_slots.json created")