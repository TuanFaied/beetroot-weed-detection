from ultralytics import YOLO 
from ultralytics.models.yolo.classify.predict import ClassificationPredictor 
import cv2

model = YOLO("best.pt")  

results = model.predict(source="0",show=True)  # Run inference

print(results.xyxy[0])  # print results