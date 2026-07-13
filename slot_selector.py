import cv2
import json

image = cv2.imread("images/parking.png")

slots = []
points = []

def click_event(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x,y])

        cv2.circle(image,(x,y),5,(0,0,255),-1)

        print("Point:",x,y)

        if len(points) % 2 == 0:
            x1,y1 = points[-2]
            x2,y2 = points[-1]

            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

            slots.append([
                [x1,y1],
                [x2,y2]
            ])

        cv2.imshow("Select Slots",image)


cv2.imshow("Select Slots",image)

cv2.setMouseCallback("Select Slots",click_event)

print("Select slots")
print("Click top-left and bottom-right of each slot")
print("Press S to save")

while True:

    key=cv2.waitKey(1)

    if key==ord("s"):

        with open("slots.json","w") as f:
            json.dump(slots,f)

        print("Saved!")
        print(slots)
        break


cv2.destroyAllWindows()