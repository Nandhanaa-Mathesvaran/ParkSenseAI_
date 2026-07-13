from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs/detect/train-4/weights/best.pt")

# Input folder
input_folder = "images"

# Output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)


# Process images
for img_name in os.listdir(input_folder):

    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):

        img_path = os.path.join(input_folder, img_name)

        img = cv2.imread(img_path)

        results = model(img, conf=0.5)

        occupied_count = 0
        empty_count = 0


        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])


                # Class names
                if cls == 0:
                    label = "empty"
                    color = (0,255,0)       # GREEN

                    empty_count += 1


                else:
                    label = "occupied"
                    color = (0,0,255)       # RED

                    occupied_count += 1


                # Draw only box
                cv2.rectangle(
                    img,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )


        # Add summary text only
        cv2.putText(
            img,
            f"Occupied: {occupied_count}",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )


        cv2.putText(
            img,
            f"Empty: {empty_count}",
            (30,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )


        save_path=os.path.join(output_folder,img_name)

        cv2.imwrite(save_path,img)

        print("Saved:",save_path)