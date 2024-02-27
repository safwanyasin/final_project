import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream with default settings
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the RealSense pipeline
pipeline.start(config)

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = depth_frame.get_distance(x, y)
        print(f"Distance to object at ({x}, {y}): {distance:.3f} meters")

# Create a window and set the callback function for mouse events
cv2.namedWindow("Depth Image")
cv2.namedWindow("Color Image")
cv2.setMouseCallback("Depth Image", on_mouse_click)

try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # Get the depth frame
        depth_frame = frames.get_depth_frame()



        # if not depth_frame or color_frame:
        if not depth_frame or not color_frame:
            continue

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Display the depth image
        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Color Image", color_image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
