import cv2
import numpy as np
import pickle

# Load the video file
video_path = "render_output.avi" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define lower and upper bounds for green color in HSV space
lower_bound = np.array([40, 50, 50])  
upper_bound = np.array([80, 255, 255])

center_points = []
heights = []


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_video_path = 'tracking_output.mp4'  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Optional: Remove noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize center point and height to None (in case no object is detected)
    center_point = None
    object_height = None

    for contour in contours:
        # Approximate the contour to detect quadrilaterals
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Detect quadrilateral shapes
            # Check if the shape is convex
            if cv2.isContourConvex(approx):
                # Get the moments of the contour to calculate the center
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = 0, 0

                # Calculate the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Set the center point and height if an object is detected
                center_point = [cx, cy]
                object_height = h  # Height of the bounding box

                # Draw the contour and the quadrilateral
                # cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)

                # Mark the center point
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Draw the bounding box and label the height
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Height: {object_height}px", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Label the detected object
                cv2.putText(frame, f"Center: ({cx}, {cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Append the center point and height (or None) for this frame
    center_points.append(center_point)
    heights.append(object_height)
    # Show the frame with detected object
    out.write(frame)
    cv2.imshow("Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the center points and heights to a pickle file
data = {'center_points': center_points, 'heights':heights }
with open("tracking_data.pickle", "wb") as f:
    pickle.dump(data, f)

# Release the video capture and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()