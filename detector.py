from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

image = cv2.imread("images/parking.png")

results = model(image, classes=[2])

annotated = results[0].plot()

cv2.imshow("Car Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()