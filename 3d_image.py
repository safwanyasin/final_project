import open3d as o3d
import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams with default settings
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the RealSense pipeline
pipeline.start(config)

# Create an Open3D visualizer
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Get the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert color and depth frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create a point cloud
        points = rs.pointcloud()
        points.map_to(color_frame)
        points.process(depth_frame)

        # Get the point cloud data
        pc_data = points.get_points()

        # Convert point cloud data to numpy array
        pc_np = np.asarray(pc_data.get_vertices())

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # Display the point cloud
        visualizer.add_geometry(pcd)
        visualizer.update_geometry()
        visualizer.poll_events()
        visualizer.update_renderer()

        # Display the color and depth images
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    visualizer.destroy_window()
    cv2.destroyAllWindows()
