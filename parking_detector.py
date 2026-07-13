import cv2
import json
import os


# Load slots
with open("auto_slots.json", "r") as f:
    slots = json.load(f)


image_folder = "images"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)



for image_name in os.listdir(image_folder):

    if not image_name.lower().endswith(
        (".png",".jpg",".jpeg")
    ):
        continue


    print("\nProcessing:", image_name)


    img = cv2.imread(
        os.path.join(image_folder,image_name)
    )


    if img is None:
        continue



    total_slots = len(slots)
    occupied_count = 0



    for slot in slots:


        x1,y1 = slot[0]
        x2,y2 = slot[1]


        crop = img[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:
            continue



        # Resize
        crop = cv2.resize(
            crop,
            (80,80)
        )


        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )


        # Blur removes parking lines
        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )


        # Dark pixel percentage
        dark_pixels = (
            (blur < 80).sum()
            /
            blur.size
        ) * 100


        # Standard deviation
        std = blur.std()



        # Occupancy decision

        if dark_pixels > 15 and std > 25:

            status = "Occupied"
            color = (0,0,255)

            occupied_count += 1


        else:

            status = "Empty"
            color = (0,255,0)



        cv2.rectangle(
            img,
            (x1,y1),
            (x2,y2),
            color,
            2
        )


        cv2.putText(
            img,
            status,
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )



    empty_count = (
        total_slots -
        occupied_count
    )


    occupancy = (
        occupied_count /
        total_slots
    )*100



    print("Total:",total_slots)
    print("Occupied:",occupied_count)
    print("Available:",empty_count)
    print("Occupancy:",round(occupancy,2))



    # Statistics box

    cv2.putText(
        img,
        f"Total:{total_slots}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        .8,
        (255,255,255),
        2
    )


    cv2.putText(
        img,
        f"Occupied:{occupied_count}",
        (20,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        .8,
        (0,0,255),
        2
    )


    cv2.putText(
        img,
        f"Available:{empty_count}",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        .8,
        (0,255,0),
        2
    )


    save = os.path.join(
        output_folder,
        image_name
    )

    cv2.imwrite(save,img)

    print("Saved:",save)


print("Done")