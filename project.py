import cv2
import torch
from ultralytics import YOLO, solutions


# import cv2

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the codec and create a VideoWriter object to save the video
# 'XVID' codec is commonly used for .avi files
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('input_video.avi', fourcc, 20.0, (640, 480))

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Write the frame to the video file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Recording', frame)

    # Stop recording if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when the recording is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# Load the YOLO model
model = YOLO("best.pt")

# Open the video file
cap = cv2.VideoCapture("input_video.avi")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define line points
line_points = [(20, 400), (600, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output6.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize Object Counter
counter = solutions.ObjectCounter(
    view_img=False,  # Set to False to prevent a separate window
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

# Define confidence threshold
confidence_threshold = 0.9
weed_class_id = 1  # Assuming 1 is the class ID for weeds
sugarbeet_class_id = 0  # Assuming 0 is the class ID for sugarbeets

# Define the quit button coordinates and size
quit_button_position = (10, 10)
quit_button_size = (100, 40)
quit_button_color = (0, 0, 255)  # Red color
quit_button_text_color = (255, 255, 255)  # White color

def quit_button_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if quit_button_position[0] <= x <= quit_button_position[0] + quit_button_size[0] and \
           quit_button_position[1] <= y <= quit_button_position[1] + quit_button_size[1]:
            global quit_program
            quit_program = True

cv2.namedWindow("YOLO Detection")
cv2.setMouseCallback("YOLO Detection", quit_button_callback)

quit_program = False

while cap.isOpened() and not quit_program:
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

    # Draw the quit button
    cv2.rectangle(im0, quit_button_position, 
                  (quit_button_position[0] + quit_button_size[0], quit_button_position[1] + quit_button_size[1]),
                  quit_button_color, -1)
    cv2.putText(im0, "QUIT", (quit_button_position[0] + 10, quit_button_position[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, quit_button_text_color, 2)

    # Show the frame
    cv2.imshow('YOLO Detection', im0)

    # Write the frame to the video
    video_writer.write(im0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
