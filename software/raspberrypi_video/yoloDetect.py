
from ultralytics import YOLO


#loads yolo model
#model = YOLO("/home/ubuntu/Downloads/thermal/LeptonModule/software/raspberrypi_video/best.pt")
model = YOLO("/home/ubuntu/Downloads/thermal/LeptonModule/software/raspberrypi_video/bestV6.pt")

#this runs and saves the YOLO model results to this variable
#you can also run the model on an entire folder of images much faster per image than a singular image
#it could also work on videos, but would be too slow on a RB3b 
results = model ("/home/ubuntu/Downloads/thermal/LeptonModule/software/raspberrypi_video/testimage.jpg")

#this shows the visuals with the bounding boxes 
results[0].show()

#this prints the bounding boxes to terminal
#print(f"bounding boxes: {results[0].boxes}")
print(f"bounding boxes: {results[0].boxes.xyxy}")

