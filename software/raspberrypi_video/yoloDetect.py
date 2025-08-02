
from ultralytics import YOLO


#loads YOLO model, change this to change update the YOLO model
model = YOLO("/home/ubuntu/Downloads/thermal/best.pt")


#this is the file the YOLO model is detecting humans in, the variable will save the results
results = model ("/home/ubuntu/Downloads/thermal/LeptonModule/software/raspberrypi_video/testimage.jpg")


#this shows the results of the YOLO model with bounding boxes visually
results[0].show()


#this prints the bounding boxes of the YOLO model results
print(f"bounding boxes: {results[0].boxes.xyxy}")


