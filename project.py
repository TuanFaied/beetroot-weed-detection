# from ultralytics import YOLO 
# from ultralytics.models.yolo.classify.predict import ClassificationPredictor 
# import cv2

# model = YOLO("best.pt")  

# results = model.predict(source="0",show=True)  # Run inference

# print(results.xyxy[0])  


# from ultralytics import YOLO
# import cv2

# # Load the YOLO model
# model = YOLO("best.pt")


# # Open a connection to the webcam
# cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# # Define the counting line (you can adjust the position)
# line_position = 200  # Y-coordinate of the counting line

# # Initialize counters
# beetroot_count = 0
# weed_count = 0

# # Function to check if an object crosses the line
# def crosses_line(y_center, line_position):
#     return y_center > line_position

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run inference on the current frame
#     results = model.predict(source=frame)

#     # Draw the counting line on the frame
#     cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)

#     # Iterate over the predictions
#     for result in results:
#         boxes = result.boxes.xyxy  # Access bounding boxes in [x1, y1, x2, y2] format
#         labels = result.boxes.cls  # Access class labels
#         for box, label in zip(boxes, labels):
#             x1, y1, x2, y2 = box
#             y_center = (y1 + y2) / 2  # Calculate the y center of the bounding box

#             if crosses_line(y_center, line_position):
#                 if label == 'sugarbeet':  # Assuming 'beetroot' is the class label
#                     beetroot_count += 1
#                 elif label == 'weed':  # Assuming 'weed' is the class label
#                     weed_count += 1

#     # Print the counts
#     print(f"Beetroot count: {beetroot_count}")
#     print(f"Weed count: {weed_count}")

#     # Display the resulting frame
#     cv2.imshow('YOLO Detection', frame)

#     # Press 'q' to stop the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()



import cv2
import torch
from ultralytics import YOLO, solutions

# Load the YOLO model
model = YOLO("best.pt")

# Open the video file
cap = cv2.VideoCapture("input2.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define line points
line_points = [(20, 400), (1200, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output6.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

# Define confidence threshold
confidence_threshold = 0.9
weed_class_id = 1  # Assuming 1 is the class ID for weeds
sugarbeet_class_id = 0  # Assuming 0 is the class ID for sugarbeets

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform tracking
    results = model.track(im0, persist=True, show=False)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.data.cpu().numpy()  # Convert boxes to numpy array

        # Iterate over each box
        for box in boxes:
            conf = box[4]  # Confidence score
            cls = int(box[5])  # Class label

            # Relabel weeds with low confidence as sugarbeet
            if conf < confidence_threshold and cls == weed_class_id:
                box[5] = sugarbeet_class_id

        results[0].boxes.data = torch.tensor(boxes).to(results[0].boxes.data.device)

    # Start counting with modified tracks
    im0 = counter.start_counting(im0, results)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

