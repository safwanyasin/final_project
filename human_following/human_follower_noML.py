import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
import pyrealsense2 as rs
from pyzbar.pyzbar import decode
import math
import serial

threshold = 0.3
top_k = 5

tolerance = 90
x_deviation = 0
y_max = 0
arr_track_data = [0, 0, 0, 0, 0, 0]
distance_to_object = 999
previous_distance = 999
theta = 0
previous_theta = 0
object_to_track = 'person'
kpx = 2
kpy = 2
kp_theta = 0.5
r = 0.06
R = 0.15
al2 = (2 * math.pi) / 3
al3 = (4 * math.pi) / 3
min_angular_velocity = 0
max_angular_velocity = 50
min_pwm_value = 150
max_pwm_value = 255
ser = ''
min_dist_threshold = 0.5

def send_pwm_values(pwm_values):
    global ser
    ser.write(bytes(str(pwm_values), 'utf-8'))
    time.sleep(1)
    while ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').strip()
        print("Received:", data)
    data = ser.readline()
    print(data)

def map_value(x, in_min, in_max, out_min, out_max):
    if x == 0.0:
        return 0
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def constrain(val, min_val, max_val):
    if val == 0.0:
        return 0
    return max(min(val, max_val), min_val)

def capture_rgb(pipeline):
    device = pipeline.get_active_profile().get_device()
    sensor = device.query_sensors()[1] 
    sensor.set_option(rs.option.exposure, 156)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if color_frame:
        rgb_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = Image.fromarray(rgb_image)

    return rgb_image

def capture_depth(pipeline):

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if depth_frame:
        depth_frame = rs.depth_frame(depth_frame)

    return depth_frame

def track_object(qr_coordinates, depth_frame):
    global x_deviation, y_max, tolerance, arr_track_data, distance_to_object, previous_distance, previous_theta, theta
    print(qr_coordinates[0], qr_coordinates[1])
    obj_x_center = qr_coordinates[0]
    obj_y_center = qr_coordinates[1]

    if(obj_x_center == 0 and obj_y_center == 0):
        print("selected object not present")
        send_pwm_values([0, 0, 0])
        arr_track_data[4] = "obj not present"
        return
    
    thread = Thread(target = move_robot)
    thread.start()
    
    arr_track_data[0] = obj_x_center
    arr_track_data[1] = obj_y_center
    x_coordinate = obj_x_center
    y_coordinate = obj_y_center
    # adj_y_coordinate =  int(y_coordinate * 1.2)
    print('x center is ', x_coordinate)
    print('y center is ', y_coordinate)
    # print('adj_y_coordinate is ', adj_y_coordinate)
    x_deviation = x_coordinate - 320
    arr_track_data[2] = x_deviation
    previous_theta = theta
    theta = (35/320) * x_deviation
    print('angle', theta)
    previous_distance = distance_to_object
    print(x_coordinate, y_coordinate)
    distance_to_object = depth_frame.get_distance(x_coordinate, y_coordinate)
    print("distance to object:", distance_to_object)
    # second_distance = depth_frame.get_distance(x_coordinate, adj_y_coordinate)
    # print("second distance to object:", second_distance)

def move_robot():
    global min_dist_threshold, x_deviation, y_max, tolerance, arr_track_data, distance_to_object, theta, previous_distance, previous_theta
    
    y = distance_to_object
    angle = math.radians(theta)

    target_distance_x = math.cos(angle) * y
    target_distance_y = math.sin(angle) * y

    distance_error_x = target_distance_x - math.cos(math.radians(previous_theta)) * previous_distance
    distance_error_y = target_distance_y - math.sin(math.radians(previous_theta)) * previous_distance
    angle_error = 0 - angle

    control_signal_x = kpx * distance_error_x
    control_signal_y = kpy * distance_error_y
    control_signal_theta = 0

    dots = [control_signal_x, control_signal_y, control_signal_theta]

    matrix_jac = [
        [-math.sin(0), math.cos(0), R],
        [-math.sin(0 + al2), math.cos(0 + al2), R],
        [-math.sin(0 + al3), math.cos(0 + al3), R]
    ]
    matrix_wheel_dots = [0, 0, 0]
    if y >= 0.5:
        for i in range(3):
            for j in range(3):
                matrix_wheel_dots[i] += matrix_jac[i][j] * dots[j]
            matrix_wheel_dots[i] /= r
    print('matrix_wheel_dots', matrix_wheel_dots)
    pwm_values = [0, 0, 0]
    for i in range(3):
        pwm_values[i] = map_value(abs(matrix_wheel_dots[i]), min_angular_velocity, max_angular_velocity, min_pwm_value, max_pwm_value)
        pwm_values[i] = constrain(pwm_values[i], min_pwm_value, max_pwm_value)

    for i in range(3):
        pwm_values[i] = int(pwm_values[i])
        if matrix_wheel_dots[i] < 0:
            pwm_values[i] = -1 * pwm_values[i]

    send_pwm_values(pwm_values)

    if(abs(x_deviation) < tolerance):
        delay1 = 0
        if(y < 0.5):
            cmd = "Stop"
            print("Red light ON, stopping the robot")
        else:
            cmd = "forward"
            print("Red light OFF, moving the robot forward")
    
    else:
        print("Red light OFF")
        if(x_deviation >= tolerance):
            cmd = "Move Left"
            delay1 = get_delay(x_deviation)
                
            print("Moving the robot left for delay:", delay1)
                
        if(x_deviation <= -1 * tolerance):
            cmd = "Move Right"
            delay1 = get_delay(x_deviation)
                
            print("Moving the robot right for delay:", delay1)

    arr_track_data[4] = cmd
    arr_track_data[5] = delay1

def get_delay(deviation):
    deviation = abs(deviation)
    if(deviation >= 250):
        d = 0.080
    elif(deviation >= 220 and deviation < 250):
        d = 0.060
    elif(deviation >= 130 and deviation < 220):
        d = 0.050
    else:
        d = 0.040
    return d

def main():
    global ser
    fps = 1
    arr_dur = [0, 0, 0]
    ser = serial.Serial('COM6', 9600)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    pipeline.start(config)
    while True:
        start_time = time.time()
        start_t0 = time.time()
        
        pil_im = capture_rgb(pipeline)
        depth_frame = capture_depth(pipeline)
       
        arr_dur[0] = time.time() - start_t0
       
        start_t1 = time.time()
        decoded_qrs = decode(pil_im)
        qr_coordinates = [0, 0]
        for qr in decoded_qrs:
            if qr.data.decode('utf-8') == 'ens492_self_following':
                qr_coordinates[0] = int(qr.rect.left + qr.rect.width/2)
                qr_coordinates[1] = int(qr.rect.top + qr.rect.height/2)

        arr_dur[1] = time.time() - start_t1
       
        start_t2 = time.time()
        track_object(qr_coordinates, depth_frame)
       
        arr_dur[2] = time.time() - start_t2
        fps = round(1.0 / (time.time() - start_time), 1)
        print("*********FPS: ", fps, "************")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()