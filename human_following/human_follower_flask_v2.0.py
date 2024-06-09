import common as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
import pyrealsense2 as rs
from pyzbar.pyzbar import decode
import math
import serial
import sys
sys.path.insert(0, '/var/www/html/earthrover')
import util as ut
import pandas as pd
from openpyxl import load_workbook

threshold=0.3 # changed from 0.2 to 0.3
top_k=5 #first five objects with prediction probability above threshhold (0.2) to be considered
#edgetpu=0

model_dir = './models/'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

tolerance= 90 
x_deviation=0
y_max=0
arr_track_data=[0,0,0,0,0,0]
distance_to_object = 999 # decide initial value for this
previous_distance = 999
theta = 0
previous_theta = 0
object_to_track='person'
kpx = 2
kpy = 2
kp_theta = 0.5
r = 0.06
R = 0.15
al2 = (2 * math.pi) / 3
al3 = (4 * math.pi) / 3
min_angular_velocity = 0
max_angular_velocity = 50
min_pwm_value = 200
max_pwm_value = 255
ser = 'there'
min_dist_threshold = 0.5
column_names = ['time', 'x', 'y', 'theta', 'dot_1', 'dot_2', 'dot_3']
df = pd.DataFrame(columns=column_names)
df.to_csv('data.csv', index=False)

#---------Flask----------------------------------------
from flask import Flask, Response
from flask import render_template

app = Flask(__name__)


@app.route('/')
def index():
    #return "Default Message"
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def add_row_to_csv(row_data, filename='data.csv'):
    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Append the new row
    new_row = pd.DataFrame([row_data], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)


def send_pwm_values(pwm_values):
    global ser

    # for pwm in pwm_values:
        # print('sending', )
        # ser.write(str(pwm).encode())
        # ser.write(b'\n')  # Send newline as a delimiter
    # ser.flush()
    pwm_values = str(pwm_values) + '~'
    # pwm_values = "[255, -128, -128]~"
    # ser2 = serial.Serial('COM6', 9600)
    # print(ser2)
    # data = ''
    # while (pwm_values[0:len(pwm_values)-1] in str(data)) == False:
    #     ser2.write(pwm_values.encode())
    #     data = ser2.readline()
    #     print(data)

    # pwm_values = "hello world" + '~'
    ser.reset_input_buffer()
    print("sending", pwm_values)
    data_to_send = pwm_values.encode('utf-8')
    # ser2.write(data_to_send)
    # data = ser2.readline()
    # print(data)
    ser.write(data_to_send)
    # while ser.in_waiting > 0:
    #     data = ser.readline().decode('utf-8').strip()
    #     print("Received:", data)
    data = ser.readline()
    print(data)
    # ser2.flush()
    # ser2.close()
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
    sensor = device.query_sensors()[1]  # Assuming the second sensor is the RGB sensor
    sensor.set_option(rs.option.exposure, 156)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if color_frame:
        rgb_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = Image.fromarray(rgb_image)

    return rgb_image


def capture_depth(pipeline):
    # decimation = rs.decimation_filter()
    # spatial = rs.spatial_filter()
    # temporal = rs.temporal_filter()
    # # disparity_to_depth = rs.disparity_transform(True)
    # depth_to_disparity = rs.disparity_transform(False)
    # hole_filling = rs.hole_filling_filter()

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if depth_frame:
        depth_frame =  depth_frame
        depth_frame = rs.depth_frame(depth_frame)
        # depth_image = temporal.process(spatial.process(depth_to_disparity.process(decimation.process(depth_frame))))
        # depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_Z16, cv2.COLOR_BGR2RGB)
        # depth_image = Image.fromarray(depth_image)
    print('width:', depth_frame.get_width())
    print('height:', depth_frame.get_height())
    return depth_frame


def track_object(objs, labels, qr_coordinates, depth_frame):
    global x_deviation, y_max, tolerance, arr_track_data, distance_to_object, previous_distance, previous_theta, theta
    
    if(len(objs) == 0):
        print("no objects to track") 
        print("Stopping the robot and turning off the red light")
        arr_track_data = [0, 0, 0, 0, 0, 0]
        return
    flag = 0
    for obj in objs:
        lbl = labels.get(obj.id, obj.id)
        if (lbl == object_to_track):
            x_min, y_min, x_max, y_max = list(obj.bbox)
            # checks if the qr is within bounding box of the person
            if (qr_coordinates[0] >= int(x_min * 640) and qr_coordinates[0] <= int(x_max * 640) and qr_coordinates[1] >= int(y_min * 480) and qr_coordinates[1] <= int(y_max * 480)):
                flag = 1
                break
        
    if(flag == 0):
        print("selected object not present")
        send_pwm_values([0, 0, 0])
        arr_track_data[4] = "obj not present"
        return

    x_diff = x_max - x_min
    y_diff = y_max - y_min
    print("x_diff: ", round(x_diff, 5))
    print("y_diff: ", round(y_diff, 5))
        
    obj_x_center = x_min + (x_diff / 2)
    obj_x_center = round(obj_x_center, 3) # center x point of the person
    
    obj_y_center = y_min + (y_diff / 2)
    obj_y_center = round(obj_y_center, 3) # center y point of the person
    
    # x_deviation = round(0.5 - obj_x_center, 3)
    y_max = round(y_max, 3)
        
    print("{", x_deviation, y_max, "}")
   
    thread = Thread(target = move_robot)
    thread.start()
    
    arr_track_data[0] = obj_x_center
    arr_track_data[1] = obj_y_center
    # arr_track_data[2] = x_deviation
    # arr_track_data[3] = y_max
    # depth_frame = capture_depth()
    x_coordinate = int(obj_x_center * 640)
    y_coordinate = int(obj_y_center * 480)
    adj_y_coordinate =  int(y_coordinate * 1.2)
    print('x center is ', x_coordinate)
    print('y center is ', y_coordinate)
    print('adj_y_coordinate is ', adj_y_coordinate)
    x_deviation = -1 * (x_coordinate - 320)
    arr_track_data[2] = x_deviation
    previous_theta = theta
    theta = (35/320) * x_deviation
    print('angle', theta)
    # getting the distance  between the center point and the camera
    previous_distance = distance_to_object
    distance_to_object = depth_frame.get_distance(x_coordinate, y_coordinate)
    print("distance to object:", distance_to_object)
    second_distance = depth_frame.get_distance(x_coordinate, adj_y_coordinate)
    print("second distance to object:", second_distance)
    # if (second_distance < distance_to_object):
    #     distance_to_object = second_distance
    arr_track_data[3] = distance_to_object

def move_robot():
    global min_dist_threshold, x_deviation, y_max, tolerance, arr_track_data, distance_to_object, theta, previous_distance, previous_theta, df
    
    print("moving robot .............!!!!!!!!!!!!!!")
    print(x_deviation, tolerance, arr_track_data)
    
    y = distance_to_object
    angle = math.radians(theta)

    target_distance_x = math.cos(angle) * y
    target_distance_y = math.sin(angle) * y

    distance_error_x = target_distance_x - math.cos(math.radians(previous_theta)) * previous_distance
    distance_error_y = target_distance_y - math.sin(math.radians(previous_theta)) * previous_distance
    angle_error = 0 - angle

    control_signal_x = kpx * distance_error_x
    control_signal_y = kpy * distance_error_y
    # control_signal_theta = kp_theta * angle_error
    control_signal_theta = 0 # only allows crab walk movement - no rotation

    dots = [control_signal_x, control_signal_y, control_signal_theta]

    # update wheel velocities
    matrix_jac = [
        [-math.sin(0), math.cos(0), R],
        [-math.sin(0 + al2), math.cos(0 + al2), R],
        [-math.sin(0 + al3), math.cos(0 + al3), R]
    ]
    print(matrix_jac)
    matrix_wheel_dots = [0, 0, 0] # wheel velocities [20, 20, -20]
    if y >= 0.5:
        for i in range(3):
            for j in range(3):
                matrix_wheel_dots[i] += matrix_jac[i][j] * dots[j]
            matrix_wheel_dots[i] /= r
    print('matrix_wheel_dots', matrix_wheel_dots)
    new_data = {
        'time': time.time(),
        'x': target_distance_x,
        'y': target_distance_y,
        'theta': angle,
        'dot_1': matrix_wheel_dots[0],
        'dot_2': matrix_wheel_dots[1],
        'dot_3': matrix_wheel_dots[2],
    }
    add_row_to_csv(new_data)
    pwm_values = [0, 0, 0]

    for i in range(3):
        pwm_values[i] = map_value(abs(matrix_wheel_dots[i]), min_angular_velocity, max_angular_velocity, min_pwm_value, max_pwm_value)
        pwm_values[i] = constrain(pwm_values[i], min_pwm_value, max_pwm_value)

    for i in range(3):
        pwm_values[i] = int(pwm_values[i])
        if matrix_wheel_dots[i] < 0:
            pwm_values[i] = -1 * pwm_values[i] # these values would be sent to arduino
    print('pwm values', pwm_values)

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
    from util import edgetpu
    global ser
    ser = serial.Serial('COM7', 9600, timeout=None)
    # data_to_send = "hello world~"
    # # time.sleep(3)
    # data_bytes = data_to_send.encode('utf-8')
    # ser.reset_input_buffer()
    # ser.write(data_bytes)
    # for i in range(200):
    #     ser.write(data_bytes)
    #     data = ser.readline()
    #     print("data", data)
    # data_bytes = data_to_send.encode('utf-8')
    # while True: 

        
    #     ser.write(data_bytes)
    #     data = ser.readline()
    #     print("data", data)
    if (edgetpu == 1):
        mdl = model_edgetpu
    else:
        mdl = model
        
    interpreter, labels = cm.load_model(model_dir, mdl, lbl, edgetpu)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    pipeline.start(config)
    
    fps = 1
    arr_dur = [0, 0, 0]
    # ser = serial.Serial('COM6', 9600)  # Update 'COM3' to the appropriate port
    while True:
        start_time = time.time()
        #----------------Capture Camera Frame-----------------
        start_t0 = time.time()
        
        pil_im = capture_rgb(pipeline)
        depth_frame = capture_depth(pipeline)
       
        arr_dur[0] = time.time() - start_t0
        #----------------------------------------------------
       
        #-------------------Inference---------------------------------
        start_t1 = time.time()
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        arr_dur[1] = time.time() - start_t1
        # add code here that would use the pil image to find a qr code and look for its contents. then pass it into the track_object function
        decoded_qrs = decode(pil_im)
        qr_coordinates = [9999, 9999]
        for qr in decoded_qrs:
            if qr.data.decode('utf-8') == 'ens492_self_following':
                # Return the coordinates of the QR code
                print('left coordinate is:', qr.rect.left)
                print('top coordinate is:', qr.rect.top)
                qr_coordinates[0] = int(qr.rect.left + qr.rect.width/2)
                qr_coordinates[1] = int(qr.rect.top + qr.rect.height/2)
        #----------------------------------------------------
       
       #-----------------other------------------------------------
        start_t2 = time.time()
        track_object(objs, labels, qr_coordinates, depth_frame)#tracking  <<<<<<<
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rgb_np_array = np.array(pil_im)
        bgr_np_array = cv2.cvtColor(rgb_np_array, cv2.COLOR_RGB2BGR)
        cv2_im = append_text_img1(bgr_np_array, objs, labels, arr_dur, arr_track_data)
        
        ret, jpeg = cv2.imencode('.jpg', cv2_im)
        pic = jpeg.tobytes()
        
        #Flask streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n\r\n')
       
        arr_dur[2] = time.time() - start_t2
        fps = round(1.0 / (time.time() - start_time), 1)
        print("*********FPS: ", fps, "************")

    cv2.destroyAllWindows()

def append_text_img1(cv2_im, objs, labels, arr_dur, arr_track_data):
    height, width, channels = cv2_im.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    global tolerance, theta
    
    #draw black rectangle on top
    cv2_im = cv2.rectangle(cv2_im, (0, 0), (width, 24), (0, 0, 0), -1)
   
    #write processing durations
    cam = round(arr_dur[0] * 1000, 0)
    inference = round(arr_dur[1] * 1000, 0)
    other = round(arr_dur[2] * 1000, 0)
    text_dur = 'Camera: {}ms   Inference: {}ms   other: {}ms'.format(cam, inference, other)
    cv2_im = cv2.putText(cv2_im, text_dur, (int(width/4) - 30, 16), font, 0.4, (255, 255, 255), 1)
   
    #write object data
    x = round(arr_track_data[0], 3)
    y = round(arr_track_data[1], 3)
    deviation = round(arr_track_data[2], 3)
    dist = round(arr_track_data[3], 3)
    cmd = arr_track_data[4]
   
    text_track = 'x: {}   y: {}   deviation: {}   dist: {}   angle: {}   cmd: {}'.format(x, y, deviation, dist, theta, cmd)
    cv2_im = cv2.putText(cv2_im, text_track, (10, height - 10), font, 0.4, (0, 0, 255), 1)
   
    #draw bounding boxes
    for obj in objs:
        x_min, y_min, x_max, y_max = list(obj.bbox)
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)
       
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
       
        #add bbox
        cv2_im = cv2.rectangle(cv2_im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        #add label
        # cv2_im = cv2.putText(cv2_im, "c", (int(x * 640), int(y * 480 * 1.2)), font, 0.4, (0, 255, 0), 1)
        cv2_im = cv2.putText(cv2_im, label, (x_min, y_min + 10), font, 0.4, (0, 255, 0), 1)

    return cv2_im

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
    
    
    