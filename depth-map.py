import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime

def save_depth_data(depth_data):
    filename = f"depth_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    np.savetxt(filename, depth_data, fmt='%d', delimiter='\t')
    print(f"Depth data saved to {filename}")

def create_depth_map():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(True)
    depth_to_disparity = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter()

    # Start streaming
    pipeline.start(config)

    try:
        save_depth_data_flag = False
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue

            processed_depth_frame = decimation.process(depth_frame)
            processed_depth_frame = depth_to_disparity.process(processed_depth_frame)
            # spatial
            processed_depth_frame = spatial.process(processed_depth_frame)
            # temporal
            processed_depth_frame = temporal.process(processed_depth_frame)
            # # disparity_to_depth
            # processed_depth_frame = disparity_to_depth.process(processed_depth_frame)
            # hole filling
            # processed_depth_frame = hole_filling.process(processed_depth_frame)
            # Convert the depth frame to a numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image_processed = np.asanyarray(processed_depth_frame.get_data())


            # Apply colormap to the depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_processed = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_processed, alpha=0.03), cv2.COLORMAP_JET)

            # Display the depth map
            cv2.imshow('Depth Map', depth_colormap)
            cv2.imshow('Processed Depth Map', depth_colormap_processed)


            # Save depth data to a text file when 'p' is pressed
            if save_depth_data_flag:
                depth_data = np.array(depth_frame.get_data())
                save_depth_data(depth_data)
                save_depth_data_flag = False

            # Set the flag to save depth data when 'p' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                save_depth_data_flag = True

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    create_depth_map()


