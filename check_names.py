import json

with open("dataset/train/_annotations.coco.json","r") as f:
    data = json.load(f)

for img in data["images"][:20]:
    print(img["file_name"])