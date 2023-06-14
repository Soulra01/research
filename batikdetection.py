import numpy as np
import cv2
from tensorflow.keras.models import load_model
# Load the YOLOv7 model
model = load_model('yolov7_model.h5')
# Create a video capture object
cap = cv2.VideoCapture(0)
# Loop over the frames in the video
while True:
    # Capture the next frame
    ret, frame = cap.read()
    # Convert the frame to a NumPy array
    frame = np.array(frame)
    # Detect the batik in the frame
    boxes, scores, classes = model.predict(frame)
    # Draw the bounding boxes on the frame
    for box, score, class_id in zip(boxes, scores, classes):
        if score > 0.5:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Batik Detection', frame)
    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object
cap.release()
# Close all open windows
cv2.destroyAllWindows()