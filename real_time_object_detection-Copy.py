# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os

# Define the file paths and confidence level
prototxt = "C:\\Users\\dell\\Downloads\\Real-Time-Object-Detection-With-OpenCV-master\\MobileNetSSD_deploy.prototxt.txt"
model = "C:\\Users\\dell\\Downloads\\Real-Time-Object-Detection-With-OpenCV-master\\MobileNetSSD_deploy.caffemodel"
confidence = 0.2

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize the video stream and FPS counter
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Define the class labels and colors
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)

    # Pass the blob through the network and obtain the predictions
    net.setInput(blob)
    predictions = net.forward()

    # Loop over the predictions
    for i in np.arange(0, predictions.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence_level = predictions[0, 0, i, 2]

        # Filter out predictions lesser than the minimum confidence level
        if confidence_level > confidence:
            # Extract the index of the class label from the 'predictions'
            idx = int(predictions[0, 0, i, 1])

            # Then compute the (x, y)-coordinates of the bounding box for the object
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Get the label with the confidence score
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence_level * 100)
            print("Object detected: ", label)

            # Draw a rectangle across the boundary of the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # Press 'q' key to break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
vs.stop()