import cv2
import json
import os

# ===============================
# Adjustable Parameters
# ===============================
BRIGHTNESS_THRESHOLD = 85
STD_THRESHOLD = 22
DARK_RATIO_THRESHOLD = 0.45

# Load parking slots
with open("auto_slots.json", "r") as f:
    slots = json.load(f)

image_folder = "images"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

image_list = os.listdir(image_folder)

for image_name in image_list:

    if image_name.lower().endswith((".png", ".jpg", ".jpeg")):

        print("\nProcessing:", image_name)

        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print("Cannot open:", image_name)
            continue

        total_slots = len(slots)
        occupied_count = 0

        for slot in slots:

            x1, y1 = slot[0]
            x2, y2 = slot[1]

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # ----------------------------
            # Feature Extraction
            # ----------------------------

            brightness = gray.mean()
            std = gray.std()

            _, binary = cv2.threshold(
                gray,
                BRIGHTNESS_THRESHOLD,
                255,
                cv2.THRESH_BINARY_INV
            )

            dark_pixels = cv2.countNonZero(binary)
            total_pixels = gray.shape[0] * gray.shape[1]

            dark_ratio = dark_pixels / total_pixels

            # ----------------------------
            # Occupancy Decision
            # ----------------------------

            occupied = False

            # Car usually has more texture
            if std > STD_THRESHOLD:
                occupied = True

            # Very dark slot
            if brightness < BRIGHTNESS_THRESHOLD and dark_ratio > DARK_RATIO_THRESHOLD:
                occupied = True

            if occupied:
                color = (0, 0, 255)
                occupied_count += 1
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

        empty_count = total_slots - occupied_count
        occupancy = occupied_count / total_slots * 100

        print("Total Slots :", total_slots)
        print("Occupied    :", occupied_count)
        print("Available   :", empty_count)
        print("Occupancy   :", round(occupancy, 2), "%")

        # ----------------------------
        # Dashboard
        # ----------------------------

        cv2.rectangle(img, (10, 10), (340, 175), (40, 40, 40), -1)

        cv2.putText(img, "ParkSense AI", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,255), 2)

        cv2.putText(img, f"Total Slots : {total_slots}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (255,255,255),2)

        cv2.putText(img, f"Occupied : {occupied_count}", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,0,255),2)

        cv2.putText(img, f"Available : {empty_count}", (20,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,255,0),2)

        cv2.putText(img, f"Occupancy : {occupancy:.1f}%", (20,160),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (255,255,255),2)

        # Legend
        cv2.rectangle(img, (370,20), (390,40), (0,0,255), -1)
        cv2.putText(img, "Occupied", (400,37),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (255,255,255),2)

        cv2.rectangle(img, (370,55), (390,75), (0,255,0), -1)
        cv2.putText(img, "Available", (400,72),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (255,255,255),2)

        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, img)

        print("Saved:", output_path)

        cv2.imshow("ParkSense AI", img)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break

cv2.destroyAllWindows()